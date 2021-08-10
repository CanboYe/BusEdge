# SPDX-FileCopyrightText: 2021 Carnegie Mellon University
#
# SPDX-License-Identifier: Apache-2.0

import glob
import os
import signal
import sys
import threading
import time
import uuid

import yaml
from core.result_manager import ResultManager
from logzero import logger


class CvatClient(object):
    def __init__(self, config):

        self.config = config

        self.task_root = os.path.join(
            self.config["root_dir"], self.config["target_name"]
        )
        self.unlabeled_dir = os.path.join(self.task_root, "unlabeled")
        self.labeled_dir = os.path.join(self.task_root, "labeled")
        self.to_label_dir = os.path.join(self.task_root, "to_label")
        self.pseudo_anno_dir = os.path.join(self.task_root, "pseudo_anno")
        self.manual_anno_dir = os.path.join(self.task_root, "manual_anno")
        self.frcnn_model_dir = os.path.join(self.task_root, "frcnn_models")
        self._init_folders()

        self.result_manager = ResultManager(
            self.labeled_dir,
            self.pseudo_anno_dir,
            self.manual_anno_dir,
            self.config["cvat"],
        )
        self.modelVersion = 0
        self.lastVesion = 0
        self.curVesion = 0

        signal.signal(signal.SIGINT, self.stop)

    def _init_folders(self):
        os.makedirs(self.unlabeled_dir, exist_ok=True)
        os.makedirs(self.labeled_dir, exist_ok=True)
        os.makedirs(self.to_label_dir, exist_ok=True)
        os.makedirs(self.pseudo_anno_dir, exist_ok=True)
        os.makedirs(self.manual_anno_dir, exist_ok=True)

    def _mv_to_label(self, src_name):
        img_name = os.path.basename(src_name)
        to_label_filename = os.path.join(self.to_label_dir, img_name)
        os.rename(src_name, to_label_filename)
        return to_label_filename

    def _load_yaml(self, yaml_dir):
        with open(yaml_dir, "r") as stream:
            try:
                return yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                logger.critical(exc)

    def _glob_unlabeled_folder(self):
        file_list = glob.glob(os.path.join(self.unlabeled_dir, "*.jpg"))
        file_list.sort(key=os.path.getmtime)
        return file_list

    def start(self):
        try:
            threading.Thread(target=self._result_thread, name="get-results").start()
        except Exception as e:
            self.stop()
            raise e

    def stop(self, *args):
        logger.info("Stop called")
        self.result_manager.terminate()
        time.sleep(5)
        for img_name in glob.glob(os.path.join(self.to_label_dir, "*.jpg")):
            os.rename(
                img_name, os.path.join(self.unlabeled_dir, os.path.basename(img_name))
            )
        logger.info("Moved images in to_label folder back to unlabeled folder")
        pid = os.getpid()
        os.kill(pid, signal.SIGKILL)

    def _result_thread(self):
        while True:
            file_list = sorted(
                glob.glob(os.path.join(self.frcnn_model_dir, "model_v_*.pth"))
            )
            if len(file_list) > 0:
                self.curVesion = int(file_list[-1].split("/")[-1].split("_")[-1][:-4])
                if self.lastVesion < self.curVesion:
                    self.modelVersion += 1
                    self.lastVesion = self.curVesion
                    logger.info(
                        "Models have been updated to version {}!".format(
                            self.modelVersion
                        )
                    )
            #             else:
            #                 self.modelVersion = 0
            file_list = self._glob_unlabeled_folder()
            if len(file_list) == 0:
                # logger.info('unlabeled folder is empty.')
                time.sleep(3)
                continue
            for filename in file_list:
                if self.result_manager._tasks_lock._value == 0:
                    logger.info("_tasks_lock")
                    while self.result_manager._tasks_lock._value == 0:
                        pass
                to_label_filename = self._mv_to_label(filename)
                self.result_manager.add((to_label_filename, self.modelVersion))
                # print(to_label_filename)
            if not self.result_manager.running:
                self.stop()
