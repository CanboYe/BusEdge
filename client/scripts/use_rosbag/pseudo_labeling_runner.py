# SPDX-FileCopyrightText: 2021 Carnegie Mellon University
#
# SPDX-License-Identifier: Apache-2.0

import torch
import torchvision

print(torch.__version__, torch.cuda.is_available())

# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger

setup_logger()

import argparse
import glob
import json
import os
import random

import cv2

# import some common libraries
import numpy as np
from annotator import PseudoAnnotator

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from matplotlib import pyplot as plt
from tqdm import tqdm


class AnnotatorRunner:
    def __init__(
        self,
        cfg_path,
        model_path,
        image_folder_path,
        categories,
        score_thres,
        not_save_prefix,
        output_dir,
    ):
        self.image_folder_path = image_folder_path
        self.not_save_prefix = not_save_prefix
        MetadataCatalog.get("dataset").thing_classes = categories
        self.metadata = MetadataCatalog.get("dataset")

        cfg = get_cfg()
        if cfg_path is not None:
            cfg.merge_from_file(cfg_path)
        else:
            cfg.merge_from_file(
                model_zoo.get_config_file(
                    "COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"
                )
            )
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = (
            score_thres  # set threshold for this model
        )
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(categories)
        cfg.MODEL.WEIGHTS = model_path
        # cfg.MODEL.WEIGHTS = "/home/cloudlet/work/gabriel-BusEdge/server/model/sign_detector/model_final.pth"
        self.predictor = DefaultPredictor(cfg)
        output_json_dir = os.path.join(
            output_dir, os.path.basename(image_folder_path) + ".json"
        )
        self.annotator = PseudoAnnotator(categories, output_json_dir)

    def run(self):
        img_list = sorted(glob.glob(os.path.join(self.image_folder_path, "*.jpg")))
        for image_path in tqdm(img_list):
            if not self.not_save_prefix:
                img_name = os.path.join(
                    "bus_data",
                    os.path.basename(self.image_folder_path),
                    os.path.basename(image_path),
                )
            else:
                img_name = os.path.basename(image_path)
            im = cv2.imread(image_path)[:, :, ::-1]
            outputs = self.predictor(im)
            instances = outputs["instances"].to("cpu")
            self.annotator.add_pseudo_annotations(img_name, instances)
        self.annotator.dump_json()


def main(args):
    annotator_runner = AnnotatorRunner(
        args.cfg,
        args.model,
        args.input,
        args.categories,
        args.thres,
        args.not_save_prefix,
        args.output,
    )
    annotator_runner.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export pseudo labeling in COCO format"
    )

    parser.add_argument(
        "-i",
        "--input",
        default="BusEdge/bus_data/cloudy_2021_04_09_16_02_cam5_filtered",
    )
    parser.add_argument("-o", "--output", default="BusEdge/pseudo_annotations")
    parser.add_argument(
        "-m",
        "--model",
        default="BusEdge/use_rosbag/model/sign_detector/new_model_final.pth",
    )
    parser.add_argument("--cfg", default=None)
    parser.add_argument("--thres", default=0.7, type=float)
    parser.add_argument(
        "-c",
        "--categories",
        nargs="+",
        type=str,
        default=[
            "stop",
            "yield",
            "do_not_enter",
            "other_regulatory",
            "other_prohibitory",
            "warning_pedestrians",
        ],
    )
    parser.add_argument(
        "--not-save-prefix",
        action="store_true",
        help="This is needed if we want to skip the dirname in annotations",
    )
    args = parser.parse_args()
    main(args)
