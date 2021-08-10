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
from busedge_protocol import busedge_pb2
from detector import AutoDetector
from detectron2.data.catalog import DatasetCatalog, MetadataCatalog
from detectron2.utils.visualizer import ColorMode, Visualizer
from gabriel_protocol import gabriel_pb2
from gabriel_server import cognitive_engine

# libs for database
from utils.db_utils import DB_Manager

# libs for exif edit
from utils.exif import set_gps_location

logger = logging.getLogger(__name__)


class AutoDetEngine(cognitive_engine.Engine):
    def __init__(
        self,
        source_name,
        use_svm,
        num_cls,
        target_name,
        visualize,
        use_livemap,
        win_pos=1,
    ):
        # -----------------------------------------------------------#
        # Initialize your model here
        # -----------------------------------------------------------#

        logger.info("Auto-Detectron-LiveMap Engine has been initilized")

        self.source_name = source_name  # sign_filter3
        self.target_name = target_name
        self.win_pos = win_pos

        MetadataCatalog.get("livemap").thing_classes = [self.target_name]
        self.metadata = MetadataCatalog.get("livemap")

        self.use_livemap = use_livemap
        if self.use_livemap:
            self.db_manager = DB_Manager()
        self.source_name = source_name
        self.visualize = visualize

        self.task_dir = os.path.join("./autoDet_tasks/", self.target_name)
        os.makedirs(self.task_dir, exist_ok=True)
        self.unlabeled_img_dir = os.path.join(self.task_dir, "unlabeled")
        os.makedirs(self.unlabeled_img_dir, exist_ok=True)
        self.pseudo_anno_dir = os.path.join(self.task_dir, "pseudo_anno")
        os.makedirs(self.pseudo_anno_dir, exist_ok=True)
        self.svc_model_dir = os.path.join(self.task_dir, "svc_models")
        os.makedirs(self.svc_model_dir, exist_ok=True)
        self.frcnn_model_dir = os.path.join(self.task_dir, "frcnn_models")
        os.makedirs(self.frcnn_model_dir, exist_ok=True)
        self.bus_sent_dir = os.path.join(self.task_dir, "bus_sent")
        os.makedirs(self.bus_sent_dir, exist_ok=True)
        self.cloudlet_sent_dir = os.path.join(self.task_dir, "cloudlet_sent")
        os.makedirs(self.cloudlet_sent_dir, exist_ok=True)

        self.detector = AutoDetector(
            self.target_name, thres=0.86, num_cls=num_cls, use_svm=use_svm
        )

        self.image_id = 0
        self.anno_id = 0
        self.pseudo_count = 0
        # self.pseudo_anno_file = open(os.path.join(self.task_dir, 'pseudo_annotations.json'), 'w')
        self.update_anno_counter = 0
        self.update_anno_frequency = 10
        self.model_version = 0

        self.predCounter = 0

    def handle(self, input_frame):
        # -----------------------------------------------------------#
        # read input from gabriel server
        # -----------------------------------------------------------#
        if input_frame.payload_type == gabriel_pb2.PayloadType.IMAGE:
            img_array = np.asarray(bytearray(input_frame.payloads[0]), dtype=np.int8)
            img = cv2.imdecode(img_array, -1)  # cv::IMREAD_UNCHANGED = -1
            extras = cognitive_engine.unpack_extras(
                busedge_pb2.EngineFields, input_frame
            )
            gps = [
                extras.gps_data.latitude,
                extras.gps_data.longitude,
                extras.gps_data.altitude,
            ]
            img_name = extras.image_filename
            camera_name = img_name.split("_")[0]
            camera_id = int(camera_name[-1])
            timestamp = img_name.split("_")[1] + "." + img_name.split("_")[2][:-4]

            # filter_results = pk.loads(extras.filter_results)
            tic = time.time()
            outputs = self.detector.predict_livemap(img)
            predict_time = time.time() - tic
            logger.info("Inference takes time {}".format(predict_time))

            if len(outputs[outputs.pred_classes == 0]):
                # Instances Class: read https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
                outputs = outputs[outputs.pred_classes == 0].to("cpu")
                cls = outputs.pred_classes
                scores = outputs.scores
                bboxes = outputs.pred_boxes
                #             masks = outputs['instances'].pred_masks

                cls = cls.numpy()
                scores = scores.numpy()
                bboxes = bboxes.tensor.numpy()
                #             masks = masks.numpy()

                v = Visualizer(
                    img,
                    metadata=self.metadata,
                    scale=0.8,
                    instance_mode=ColorMode.IMAGE,  # remove the colors of unsegmented pixels
                )
                v = v.draw_instance_predictions(outputs)

                if self.visualize:
                    window_name = "Cloudlet Results: " + self.target_name
                    width, height = 480, 270
                    cv2.namedWindow(
                        window_name,
                        cv2.WINDOW_NORMAL
                        | cv2.WINDOW_KEEPRATIO
                        | cv2.WINDOW_GUI_NORMAL,
                    )
                    cv2.moveWindow(
                        window_name,
                        2000 + width,
                        150 + (height + 30) * self.win_pos,
                    )
                    cv2.resizeWindow(window_name, width, height)
                    cv2.imshow(window_name, v.get_image()[:, :, ::-1])
                    cv2.waitKey(1)

                    # cv2.namedWindow('Sign Detector', 0)
                    # cv2.imshow('Sign Detector', v.get_image()[:, :, ::-1])
                    # cv2.waitKey(1)

                self.predCounter += 1
                frame_for_web = cv2.resize(v.get_image()[:, :, ::-1], (640, 360))
                # t = time.localtime()
                # current_time = time.strftime("_%m_%d_%H_%M_%S", t)
                # img_name = self.source_name + str(current_time) + '.jpg'
                det_img_folder = os.path.join(
                    time.strftime("%Y_%m_%d/"), self.target_name
                )
                os.makedirs("./images/" + det_img_folder, exist_ok=True)
                det_img_dir = os.path.join(det_img_folder, img_name)
                cv2.imwrite("./images/" + det_img_dir, frame_for_web)

                if self.use_livemap:
                    for i in range(bboxes.shape[0]):
                        inserted_flag = self.db_manager.select_and_insert(
                            gps[0],
                            gps[1],
                            gps[2],
                            self.predCounter,
                            det_img_dir,
                            bboxes[i, :].flatten(),
                            self.target_name,
                            camera_id,
                            timestamp,
                            dist_thres=0.5,
                        )

                logger.info(
                    "{} objects are detected, GPS: ({:.4f},{:.4f})".format(
                        bboxes.shape[0], gps[0], gps[1]
                    )
                )

        else:
            status = gabriel_pb2.ResultWrapper.Status.WRONG_INPUT_FORMAT
            return cognitive_engine.create_result_wrapper(status)

        result_wrapper = gabriel_pb2.ResultWrapper()
        result_wrapper.result_producer_name.value = self.source_name
        result_wrapper.status = gabriel_pb2.ResultWrapper.Status.SUCCESS

        return result_wrapper
