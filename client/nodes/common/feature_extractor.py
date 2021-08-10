# SPDX-FileCopyrightText: 2021 Carnegie Mellon University
#
# SPDX-License-Identifier: Apache-2.0

# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger

setup_logger()

import glob
import os
import pickle as pk
import time

import cv2
import detectron2.data.transforms as T

# import some common libraries
import numpy as np

# import pytorch libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
from d2go.model_zoo import model_zoo
from d2go.runner import GeneralizedRCNNRunner

# import some common detectron2 utilities
# from detectron2 import model_zoo
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import (
    DatasetCatalog,
    DatasetMapper,
    MetadataCatalog,
    build_detection_test_loader,
    build_detection_train_loader,
)
from detectron2.data.datasets import register_coco_instances
from detectron2.data.samplers import InferenceSampler, TrainingSampler
from detectron2.modeling import build_model
from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputs, fast_rcnn_inference
from detectron2.structures.image_list import ImageList
from detectron2.utils.events import EventStorage
from detectron2.utils.visualizer import ColorMode, Visualizer


class FeatExtractor:
    def __init__(
        self,
        device="cpu",
        feature_extractor="FRCNN_FBNet",
    ):
        self.device = device
        self.feature_extractor = feature_extractor

        self.cfg = self._get_cfg()
        self.model = self._get_model()
        self.model.eval()

        MetadataCatalog.get("binary_classes").thing_classes = ["positive"]
        self.metadata = MetadataCatalog.get("binary_classes")

        print(self.cfg)

    def _get_normalizer(self):
        assert len(self.cfg.MODEL.PIXEL_MEAN) == len(self.cfg.MODEL.PIXEL_STD)
        num_channels = len(self.cfg.MODEL.PIXEL_MEAN)
        pixel_mean = (
            torch.Tensor(self.cfg.MODEL.PIXEL_MEAN)
            .to(self.device)
            .view(num_channels, 1, 1)
        )
        pixel_std = (
            torch.Tensor(self.cfg.MODEL.PIXEL_STD)
            .to(self.device)
            .view(num_channels, 1, 1)
        )
        normalizer = lambda x: (x - pixel_mean) / pixel_std
        return normalizer

    def _get_data_augmentations(self):
        augs = T.ResizeShortestEdge(
            [self.cfg.INPUT.MIN_SIZE_TEST, self.cfg.INPUT.MIN_SIZE_TEST],
            self.cfg.INPUT.MAX_SIZE_TEST,
        )
        return augs

    def _get_cfg(self):
        if self.feature_extractor == "FRCNN_FBNet":
            self.d2go_runner = GeneralizedRCNNRunner()
            cfg = self.d2go_runner.get_default_cfg()
            cfg.merge_from_file(
                model_zoo.get_config_file("faster_rcnn_fbnetv3a_C4.yaml")
            )
            cfg.MODEL_EMA.ENABLED = False
            cfg.DATALOADER.NUM_WORKERS = 2
            # cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("faster_rcnn_fbnetv3a_C4.yaml")  # Let training initialize from model zoo
            cfg.MODEL.WEIGHTS = "/home/albert/workspace/d2go/demo/froze_all_SVM_One_enhance_recall_ablation/network_output/model_final.pth"
        elif self.feature_extractor == "FRCNN_R50_FPN":
            cfg = get_cfg()
            cfg.merge_from_file(
                model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
            )
            cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
                "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
            )
        else:
            raise NotImplementedError(
                "unknown feature extractor " + self.feature_extractor
            )
        cfg.INPUT.MAX_SIZE_TRAIN = 1280
        cfg.INPUT.MIN_SIZE_TRAIN = (224, 256, 360, 480, 540)

        cfg.INPUT.MAX_SIZE_TEST = 1280
        cfg.INPUT.MIN_SIZE_TEST = 480
        cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION = 0.95
        cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.3
        cfg.TEST.DETECTIONS_PER_IMAGE = 20
        cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS = False

        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
        cfg.MODEL.DEVICE = self.device

        # Following FSCE to provide more frontground proposals
        cfg.MODEL.RPN.PRE_NMS_TOPK_TEST = 1000
        cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 500
        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
        cfg.MODEL.ROI_HEADS.IOU_THRESHOLD = 0.4
        return cfg

    def _get_model(self):
        if self.feature_extractor == "FRCNN_R50_FPN":
            model = build_model(self.cfg)
            checkpointer = DetectionCheckpointer(model)
            checkpointer.load(self.cfg.MODEL.WEIGHTS)
            return model
        elif self.feature_extractor == "FRCNN_FBNet":
            model = self.d2go_runner.build_model(self.cfg)
            checkpointer = DetectionCheckpointer(model)
            checkpointer.load(self.cfg.MODEL.WEIGHTS)
            return model
        else:
            raise NotImplementedError(
                "unknown feature extractor " + self.feature_extractor
            )

    def extract_box_features(self, im, train=False):
        """
        Args:
            im (np.array): a image in RGB format
        """
        height, width = im.shape[:2]
        image = self._get_data_augmentations().get_transform(im).apply_image(im)
        image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
        batched_inputs = [{"height": height, "width": width, "image": image}]

        with torch.no_grad():
            # forward
            # Normalize, pad and batch the input images. (Preprocess_image)
            images = [x["image"].to(self.device) for x in batched_inputs]
            images = [self._get_normalizer()(x) for x in images]
            images = ImageList.from_tensors(
                images, self.model.backbone.size_divisibility
            )
            features = self.model.backbone(images.tensor)
            # print('features shape:', features["trunk3"].shape)
            proposals, _ = self.model.proposal_generator(images, features)
            # print('proposal num per img:', proposals[0].objectness_logits.shape)

            if train:
                targets = [d["instances"].to(self.device) for d in batched_inputs]
                proposals = self.model.roi_heads.label_and_sample_proposals(
                    proposals, targets
                )

            box_features = self.model.roi_heads.box_pooler(
                [features[f] for f in self.cfg.MODEL.ROI_HEADS.IN_FEATURES],
                [x.proposal_boxes for x in proposals],
            )
            box_features = self.model.roi_heads.box_head(box_features)
        # print('box_feature_shape: ', box_features.shape)
        return box_features, proposals

    def extract_features(self, im):
        """
        Args:
            im (np.array): a image in RGB format
        """
        height, width = im.shape[:2]
        image = self._get_data_augmentations().get_transform(im).apply_image(im)
        image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
        batched_inputs = [{"height": height, "width": width, "image": image}]

        with torch.no_grad():
            images = [x["image"].to(self.device) for x in batched_inputs]
            images = [self._get_normalizer()(x) for x in images]
            images = ImageList.from_tensors(
                images, self.model.backbone.size_divisibility
            )
            features = self.model.backbone(images.tensor)

            features = features["trunk3"].squeeze().to("cpu").detach().numpy()

        # print("img_feat: ", features.keys())
        # for k in list(features.keys()):
        #     print("features_", k, ":", features[k].shape)
        # features_trunk0 : torch.Size([1, 16, 240, 427])
        # features_trunk1 : torch.Size([1, 24, 120, 214])
        # features_trunk2 : torch.Size([1, 32, 60, 107])
        # features_trunk3 : torch.Size([1, 112, 30, 54])
        #  (batch, channels, height, width)

        return features

    def _visualize_results(self, raw_image, result_show, save_img_dir, metadata):
        v = Visualizer(
            raw_image, metadata=metadata, scale=1.0, instance_mode=ColorMode.IMAGE
        )
        v = v.draw_instance_predictions(result_show.to("cpu"))
        window_name = "filtered results"
        cv2.namedWindow(
            window_name,
            cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_NORMAL,
        )
        cv2.imshow(window_name, v.get_image()[:, :, ::-1])
        cv2.waitKey(1)
        if save_img_dir is not None:
            os.makedirs(os.path.dirname(save_img_dir), exist_ok=True)
            cv2.imwrite(save_img_dir, v.get_image()[:, :, ::-1])

    def _filter_small(self, boxes, scores, thres_area=1000):
        filtered_boxes = []
        filtered_scores = []
        results_len = scores.shape[0]

        for i in range(results_len):
            box = boxes[i]
            score = scores[i]
            area = (box[2] - box[0]) * (box[3] - box[1])
            if area > thres_area:
                filtered_boxes.append(box.reshape((1, 4)))
                filtered_scores.append(score)
            else:
                print("filtered small box")
        if len(filtered_scores) > 0:
            filtered_boxes = np.concatenate(filtered_boxes, axis=0)
            filtered_scores = np.array(filtered_scores)
            return filtered_boxes, filtered_scores
        else:
            return np.array([[]]), np.array([])
