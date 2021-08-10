# SPDX-FileCopyrightText: 2021 Carnegie Mellon University
#
# SPDX-License-Identifier: Apache-2.0

import io
import json
import logging
import os
import pickle as pk
import time

import cv2
import numpy as np
import torch
from detector import AutoDetector
from gabriel_protocol import gabriel_pb2
from gabriel_server import cognitive_engine
from protocol import busedge_autodet_pb2

logger = logging.getLogger(__name__)


class AutoDetEngine(cognitive_engine.Engine):
    def __init__(self, target_name, use_svm, num_cls):
        # -----------------------------------------------------------#
        # Initialize your model here
        # -----------------------------------------------------------#

        logger.info("Auto-Detectron Engine has been initilized")

        self.target_name = target_name
        self.source_name = "autoDet_" + self.target_name
        self.task_dir = os.path.join("./autoDet_tasks/", self.target_name)
        os.makedirs(self.task_dir, exist_ok=True)
        self.unlabeled_img_dir = os.path.join(self.task_dir, "unlabeled")
        os.makedirs(self.unlabeled_img_dir, exist_ok=True)
        self.pseudo_anno_dir = os.path.join(self.task_dir, "pseudo_anno")
        os.makedirs(self.pseudo_anno_dir, exist_ok=True)
        self.svc_model_dir = os.path.join(self.task_dir, "svc_models_bus")
        os.makedirs(self.svc_model_dir, exist_ok=True)
        self.frcnn_model_dir = os.path.join(self.task_dir, "frcnn_models")
        os.makedirs(self.frcnn_model_dir, exist_ok=True)
        self.bus_sent_dir = os.path.join(self.task_dir, "bus_sent")
        os.makedirs(self.bus_sent_dir, exist_ok=True)
        self.cloudlet_sent_dir = os.path.join(self.task_dir, "cloudlet_sent")
        os.makedirs(self.cloudlet_sent_dir, exist_ok=True)

        self.detector = AutoDetector(
            self.target_name, thres=0.7, num_cls=num_cls, use_svm=use_svm
        )

        self._reset_pseudo_annotations()
        self.image_id = 0
        self.anno_id = 0
        self.pseudo_count = 0
        # self.pseudo_anno_file = open(os.path.join(self.task_dir, 'pseudo_annotations.json'), 'w')
        self.update_anno_counter = 0
        self.update_anno_frequency = 10
        self.model_version = 0
        self.save_raw = True

    def handle(self, input_frame):
        # -----------------------------------------------------------#
        # read input from gabriel server
        # -----------------------------------------------------------#
        if input_frame.payload_type == gabriel_pb2.PayloadType.IMAGE:
            img_array = np.asarray(bytearray(input_frame.payloads[0]), dtype=np.int8)
            img = cv2.imdecode(img_array, -1)  # cv::IMREAD_UNCHANGED = -1
            extras = cognitive_engine.unpack_extras(
                busedge_autodet_pb2.EngineFields, input_frame
            )
            gps = [
                extras.gps_data.latitude,
                extras.gps_data.longitude,
                extras.gps_data.altitude,
            ]
            img_name = extras.image_filename

            if self.save_raw:
                save_dir = os.path.join(self.bus_sent_dir, str(self.model_version))
                os.makedirs(save_dir, exist_ok=True)
                raw_img_dir = os.path.join(save_dir, img_name)
                cv2.imwrite(raw_img_dir, img[:, :, ::-1])

            # filter_results = pk.loads(extras.filter_results)
            tic = time.time()
            save_img_dir = os.path.join(
                self.cloudlet_sent_dir, str(self.model_version), img_name
            )
            boxes, scores = self.detector.predict(img, save_img_dir)
            predict_time = time.time() - tic
            logger.debug("Inference takes time {}".format(predict_time))

            boxes = boxes.astype(float)
            boxes = np.around(boxes, decimals=3)
            scores = scores.astype(float)
            scores = np.around(scores, decimals=3)

            if scores.shape[0] != 0:
                #                 if self.save_raw:
                #                     save_dir = os.path.join(self.cloudlet_sent_dir, str(self.model_version))
                #                     os.makedirs(save_dir, exist_ok=True)
                #                     raw_img_dir = os.path.join(save_dir, img_name)
                #                     cv2.imwrite(raw_img_dir, img[:, :, ::-1])
                self.add_pseudo_annotations(
                    boxes, img_name, img.shape[0], img.shape[1], scores
                )
                self.update_anno_counter += 1
                if self.update_anno_counter >= self.update_anno_frequency:
                    self.update_pseudo_annotations(img_name)
                    self.update_anno_counter = 0

                unlabeled_img_dir = os.path.join(self.unlabeled_img_dir, img_name)
                cv2.imwrite(unlabeled_img_dir, img[:, :, ::-1])
                logger.info("Saved an image with its psuedo annotation.")

        elif input_frame.payload_type == gabriel_pb2.PayloadType.OTHER:
            logger.debug("BusEdge client is requesting model.")

        else:
            status = gabriel_pb2.ResultWrapper.Status.WRONG_INPUT_FORMAT
            return cognitive_engine.create_result_wrapper(status)

        result_wrapper = gabriel_pb2.ResultWrapper()
        result_wrapper.result_producer_name.value = self.source_name
        result_wrapper.status = gabriel_pb2.ResultWrapper.Status.SUCCESS
        # -----------------------------------------------------------#
        # if you need to send the results back to the client
        # -----------------------------------------------------------#
        if self.check_model_update():
            model_bytes = self.load_svm_model_bytes()
            result = gabriel_pb2.ResultWrapper.Result()
            result.payload_type = gabriel_pb2.PayloadType.OTHER
            result.payload = model_bytes
            result_wrapper.results.append(result)
            logger.info("Updated model has been sent back to client.")
        else:
            pass

        return result_wrapper

    def _reset_pseudo_annotations(self):
        self.coco_anno_dict = {
            "info": [],
            "licenses": [],
            "categories": [{"id": 1, "name": "positive", "supercategory": ""}],
            "images": [],
            "annotations": [],
        }

    def add_pseudo_annotations(self, boxes, img_name, height, width, scores):
        self.image_id += 1
        image_dict = {
            "id": self.image_id,
            "width": width,
            "height": height,
            "file_name": img_name,
        }
        self.coco_anno_dict["images"].append(image_dict)
        anno_dicts = []
        for i in range(boxes.shape[0]):
            self.anno_id += 1
            box = boxes[i]
            area = box[2] * box[3]
            box_coco = [box[0], box[1], box[2] - box[0], box[3] - box[1]]
            anno_dict = {
                "id": self.anno_id,
                "image_id": self.image_id,
                "category_id": 1,
                "area": area,
                "bbox": box_coco,
                "iscrowd": 0,
                "score": scores[i],
            }
            anno_dicts.append(anno_dict)
        self.coco_anno_dict["annotations"].extend(anno_dicts)

    def update_pseudo_annotations(self, img_name):
        # self.pseudo_anno_file.seek(0)
        # json.dump(self.coco_anno_dict, self.pseudo_anno_file)
        # self.pseudo_anno_file.truncate()
        self.pseudo_count += 1
        pseudo_anno_file = open(
            os.path.join(
                self.pseudo_anno_dir, "pseudo_anno_{}.json".format(self.pseudo_count)
            ),
            "w",
        )
        json.dump(self.coco_anno_dict, pseudo_anno_file)
        self._reset_pseudo_annotations()

    def check_model_update(self):
        flag_file = os.path.join(self.svc_model_dir, "update_flag")
        if not os.path.exists(flag_file):
            return False
        with open(flag_file, "r+") as f:
            flag = int(f.read().strip())
            if flag >= 1:
                f.seek(0)
                f.write("0")
                f.truncate()
                self.model_version = flag
                return True
            else:
                return False

    def load_svm_model_bytes(self):
        model_dir = os.path.join(
            self.svc_model_dir, "svc_model_{}.pkl".format(self.model_version)
        )
        with open(model_dir, "rb") as file:
            model_bytes = file.read()
        logger.info(
            "SVC model has been updated to version {}".format(self.model_version)
        )
        return model_bytes

    def load_cnn_model_bytes(self, model_dir):
        state_dict = torch.load(model_dir)
        model_bytes = io.BytesIO()
        torch.save(state_dict, model_bytes)
        model_bytes.seek(0)
        return model_bytes
