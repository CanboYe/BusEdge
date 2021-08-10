# SPDX-FileCopyrightText: 2021 Carnegie Mellon University
#
# SPDX-License-Identifier: Apache-2.0

import json
import logging
import os
import shutil
import sys
import threading
import time
from http.client import HTTPConnection
from threading import Timer

import numpy as np
import requests
from core.cvat import CLI, CVAT_API_V1
from logzero import logger


class RepeatedTimer(object):
    def __init__(self, interval, function, *args, **kwargs):
        self._timer = None
        self.interval = interval
        self.function = function
        self.args = args
        self.kwargs = kwargs
        self.is_running = False
        self.start()

    def _run(self):
        self.is_running = False
        self.start()
        self.function(*self.args, **self.kwargs)

    def start(self):
        if not self.is_running:
            self._timer = Timer(self.interval, self._run)
            self._timer.start()
            self.is_running = True

    def stop(self):
        self._timer.cancel()
        self.is_running = False


class ResultManager(object):
    def __init__(self, train_path, pseudo_anno_path, manual_anno_path, cvat_config):
        task_config = cvat_config["tasks"]
        self.train_path = train_path
        self.results = []
        self.results_len = task_config["length"]
        self.results_wait = task_config["time"]
        self.pseudo_anno_path = pseudo_anno_path
        self.manual_anno_path = manual_anno_path

        self.pending_tasks = {}
        self.finished_ids = set()
        self.seen_results = set()
        self.total_tasks = max(int(task_config.get("pending", "3")), 1)
        self.max_length = self.total_tasks * self.results_len
        self.curr_task_id = 0

        self.results_lock = threading.Lock()
        self.train_lock = threading.Lock()
        self._tasks_lock = threading.Semaphore(self.total_tasks)
        self.is_full_flag = False

        self.running = True

        # self.labels = [{'attributes': [], 'name': 'positive'}]
        with open(task_config["label_path"]) as f:
            self.labels = json.load(f)

        with requests.Session() as session:
            self.api = CVAT_API_V1(
                "{}:{}".format(cvat_config["host"], cvat_config["port"])
            )
            self.cli = CLI(
                session, self.api, [cvat_config["user"], cvat_config["password"]]
            )

        self._status_monitor = RepeatedTimer(15, self._monitor_tasks)
        self._result_check = RepeatedTimer(2, self._check_results)
        self.time_start = time.time()
        self.terminate_counter = 0
        self.termination_limit = 3

    def on_running(func):
        def wrapper(self, *args, **kwargs):
            if self.running:
                return func(self, *args, **kwargs)

        return wrapper

    def is_full(self):
        if len(self.results) >= self.max_length:
            self.is_full_flag = True
            return True
        self.is_full_flag = False
        return False

    @on_running
    def add(self, result):
        with self.results_lock:
            result_id = result[0]
            if os.path.basename(result_id) not in self.seen_results:
                self.results.append(result)
                logger.info("a result is added")
        while self.is_full():
            time.sleep(3)
            logger.critical("is full!")
            continue

    @on_running
    def _check_length(self):
        with self.results_lock:
            # logger.info('len_results = {}, < {}?'.format(len(self.results), self.results_len))
            if (len(self.results) < self.results_len) and (
                round(time.time() - self.time_start) < self.results_wait
            ):
                return
            self._create_task()
        if self.terminate_counter >= self.termination_limit:
            self.running = False

    def all_task_locks_free(self):
        return self._tasks_lock._value == self.total_tasks

    def all_task_locks_acquired(self):
        return self._tasks_lock._value == 0

    @on_running
    def _create_task(self):
        logger.info("Now trying to create new task")
        if not self.results:
            time_lapsed = round(time.time() - self.time_start)
            if time_lapsed > self.results_wait:
                self.terminate_counter += 1
                logger.debug(
                    "Waiting to terminate {}/{}".format(
                        self.terminate_counter, self.termination_limit
                    )
                )
            return
        self._tasks_lock.acquire()
        task_results = self.results[: self.results_len]
        self.results = self.results[self.results_len :]
        try:
            result_ids, versions = zip(*task_results)
            last_image_name = os.path.basename(result_ids[-1])
            result_ids = set(result_ids)
            # result_ids = sorted(set(result_ids))
            model_version = min(versions)
            self.curr_task_id += 1
            task_name = f"task-{self.curr_task_id}-model-{model_version}"
            result_ids = [
                p
                for p in result_ids
                if os.path.exists(p) and (os.path.basename(p) not in self.seen_results)
            ]

            [self.seen_results.add(os.path.basename(p)) for p in result_ids]
            if not result_ids:
                self.terminate_counter += 1
                return
            with self.train_lock:
                pseudo_anno_file = os.path.join(
                    self.pseudo_anno_path,
                    "pseudo_anno_{}.json".format(self.curr_task_id - 1),
                )
                if os.path.exists(pseudo_anno_file):
                    task_id = self.cli.tasks_create(
                        task_name,
                        self.labels,
                        result_ids,
                        annotation_path=pseudo_anno_file,
                    )
                else:  # for initialization
                    logger.info(
                        "Cannot find pseudo annotations, creating tasks for initialization."
                    )
                    # init_label = self.labels[0].copy()
                    # init_label["attributes"] = []
                    task_id = self.cli.tasks_create(
                        task_name, self.labels, result_ids, annotation_path=""
                    )
                self.pending_tasks[task_id] = result_ids
            self.time_start = time.time()
        except (
            requests.exceptions.HTTPError,
            requests.exceptions.ConnectionError,
            requests.exceptions.RequestException,
        ) as e:
            self._tasks_lock.release()
            logger.critical(e)

    def _monitor_tasks(self):
        # logger.info('get in _monitor_tasks')
        self._check_status()

    def _check_results(self):
        # logger.info('get in _check_results')
        self._check_length()

    def _get_completed_tasks(self, task_ids):
        stats = self.cli.tasks_status(task_ids)
        completed_stats = filter(lambda x: x["status"] == "completed", stats)
        completed_ids = [s["id"] for s in completed_stats]
        completed_ids = list(set(completed_ids).difference(self.finished_ids))
        return completed_ids

    @on_running
    def _check_status(self):
        with self.train_lock:
            task_ids = self.pending_tasks.keys()
            if not task_ids:
                return
            try:
                for i in self._get_completed_tasks(task_ids):
                    task_id, anno_dict = self.cli.tasks_dump(i)
                    self._add_train(task_id, anno_dict)
                    # updating task start time to prevent a new task
                    # creation immediatelly after releasing lock
                    self.time_start = time.time()
                    self._tasks_lock.release()

            except (
                requests.exceptions.HTTPError,
                requests.exceptions.ConnectionError,
                requests.exceptions.RequestException,
            ) as e:
                logger.critical(e)

    def _add_train(self, task_id, anno_dict, positive_ids=None):
        if (task_id not in self.pending_tasks) or (task_id in self.finished_ids):
            return

        task_images = self.pending_tasks[task_id]
        del self.pending_tasks[task_id]

        all_results = np.array(task_images)

        last_img_name = os.path.basename(all_results[-1])
        # anno_filename = os.path.join(
        #     self.manual_anno_path, "manual_anno_{}.json".format(last_img_name[:-4])
        # )
        anno_filename = os.path.join(
            self.manual_anno_path, "cvat_annotations_{}.json".format(self.curr_task_id)
        )
        with open(anno_filename, "w") as fp:
            json.dump(anno_dict, fp)

        if positive_ids is not None:
            positive_ids = np.asarray(positive_ids, dtype=np.int32)
            mask_ids = np.zeros(all_results.size, dtype=bool)
            mask_ids[positive_ids] = True
            positives = all_results[mask_ids]
            negatives = all_results[~mask_ids]

            for positive in positives:
                label = "1"
                dst_path = os.path.join(
                    self.train_path, label, os.path.basename(positive)
                )
                if os.path.exists(positive):
                    shutil.move(positive, dst_path)
            for negative in negatives:
                label = "0"
                dst_path = os.path.join(
                    self.train_path, label, os.path.basename(negative)
                )
                if os.path.exists(negative):
                    shutil.move(negative, dst_path)
        else:
            for labeled_img in all_results:
                dst_path = os.path.join(self.train_path, os.path.basename(labeled_img))
                if os.path.exists(labeled_img):
                    shutil.move(labeled_img, dst_path)

        [self.seen_results.remove(os.path.basename(p)) for p in all_results]

        self.finished_ids.add(task_id)
        time.sleep(5)

    def terminate(self):
        self.running = False
        self._result_check.stop()
        self._status_monitor.stop()
        for _ in range(self.total_tasks):
            self._tasks_lock.release()
        # import psutil
        # current_process_pid = psutil.Process().pid


#         import signal
#         pid = os.getpid()
#         os.kill(pid, signal.SIGKILL)
