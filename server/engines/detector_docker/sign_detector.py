# SPDX-FileCopyrightText: 2021 Carnegie Mellon University
#
# SPDX-License-Identifier: Apache-2.0

import logging
import os

# libs for detector
import random
import time

import cv2
import imageio
import numpy as np
from busedge_protocol import busedge_pb2
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data.catalog import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances

# import cv2
from detectron2.engine import DefaultTrainer
from detectron2.engine.defaults import DefaultPredictor
from detectron2.utils.visualizer import ColorMode, Visualizer
from gabriel_protocol import gabriel_pb2
from gabriel_server import cognitive_engine

# libs for database
from utils.db_utils import DB_Manager

# libs for exif edit
from utils.exif import set_gps_location

logger = logging.getLogger(__name__)

#                 0      1            2         3: black and white    4:red circle               5
CATEGORIES = [
    "stop",
    "yield",
    "do_not_enter",
    "other_regulatory",
    "other_prohibitory",
    "warning_pedestrians",
]


class SignDetectorEngine(cognitive_engine.Engine):
    def __init__(self, source_name, visualize, save_raw, use_livemap):
        MetadataCatalog.get("empty_dataset").thing_classes = CATEGORIES
        self.mapillary_metadata = MetadataCatalog.get("empty_dataset")
        cfg = get_cfg()
        cfg.merge_from_file(
            model_zoo.get_config_file(
                "COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"
            )
        )
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.80  # set threshold for this model
        cfg.MODEL.WEIGHTS = "./model/sign_detector/model_final.pth"
        # cfg.MODEL.DEVICE = 'cpu'
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(CATEGORIES)
        print("loading from: {}".format(cfg.MODEL.WEIGHTS))
        self.predictor = DefaultPredictor(cfg)
        self.predCounter = 0
        logger.info("Detector has been initilized")

        self.use_livemap = use_livemap
        if self.use_livemap:
            self.db_manager = DB_Manager()
        self.source_name = source_name
        self.visualize = visualize
        self.save_raw = save_raw

    def handle(self, input_frame):
        if input_frame.payload_type != gabriel_pb2.PayloadType.IMAGE:
            status = gabriel_pb2.ResultWrapper.Status.WRONG_INPUT_FORMAT
            return cognitive_engine.create_result_wrapper(status)
        extras = cognitive_engine.unpack_extras(busedge_pb2.EngineFields, input_frame)

        gps = [
            extras.gps_data.latitude,
            extras.gps_data.longitude,
            extras.gps_data.altitude,
        ]
        img_name = extras.image_filename
        camera_name = img_name.split("_")[0]
        camera_id = int(camera_name[-1])
        timestamp = img_name.split("_")[1] + "." + img_name.split("_")[2][:-4]
        # -----------------------------------------------------------#
        # Run detector on client input
        # -----------------------------------------------------------#
        show_flag = True

        img_array = np.asarray(bytearray(input_frame.payloads[0]), dtype=np.int8)
        img = cv2.imdecode(img_array, -1)  # cv::IMREAD_UNCHANGED = -1,
        if self.save_raw:
            raw_img_folder = "./raw_images/" + time.strftime("%Y_%m_%d/")
            os.makedirs(raw_img_folder, exist_ok=True)
            raw_img_dir = raw_img_folder + img_name
            cv2.imwrite(raw_img_dir, img[:, :, ::-1])
            set_gps_location(
                raw_img_dir,
                extras.gps_data.latitude,
                extras.gps_data.longitude,
                extras.gps_data.altitude,
            )
            if self.use_livemap:
                self.db_manager.insert_rec_images(
                    gps[0],
                    gps[1],
                    gps[2],
                    raw_img_dir,
                    [
                        0,
                    ],
                    camera_id,
                    timestamp,
                )

        time_start = time.time()
        outputs = self.predictor(img)
        time_end = time.time()
        time_cost = time_end - time_start
        logger.info("Received an image, detection takes {} seconds".format(time_cost))

        if len(outputs["instances"]):
            # Instances Class: read https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
            outputs = outputs["instances"].to("cpu")
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
                metadata=self.mapillary_metadata,
                scale=0.8,
                instance_mode=ColorMode.IMAGE,  # remove the colors of unsegmented pixels
            )
            v = v.draw_instance_predictions(outputs)

            if self.visualize:
                window_name = "Cloudlet Results: " + camera_name
                width, height = 320, 180
                cv2.namedWindow(
                    window_name,
                    cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_NORMAL,
                )
                if self.predCounter < 10:
                    cv2.moveWindow(
                        window_name,
                        20 + (camera_id - 1) * width + 20,
                        300 + (height + 30) * 2 + 100,
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
            det_img_folder = time.strftime("%Y_%m_%d/")
            os.makedirs("./images/" + det_img_folder, exist_ok=True)
            det_img_dir = det_img_folder + img_name
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
                        CATEGORIES[cls[i]],
                        camera_id,
                        timestamp,
                        dist_thres=0.5,
                    )

            logger.info(
                "{} objects are detected, GPS: ({:.4f},{:.4f})".format(
                    bboxes.shape[0], gps[0], gps[1]
                )
            )
        # -----------------------------------------------------------#

        # result = gabriel_pb2.ResultWrapper.Result()
        # result.payload_type = gabriel_pb2.PayloadType.IMAGE
        # result.engine_name = self.ENGINE_NAME
        # result.payload = input_frame.payloads[0]
        # result_wrapper.results.append(result)

        result_wrapper = gabriel_pb2.ResultWrapper()
        result_wrapper.result_producer_name.value = self.source_name
        result_wrapper.status = gabriel_pb2.ResultWrapper.Status.SUCCESS

        return result_wrapper
