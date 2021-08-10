# SPDX-FileCopyrightText: 2021 Carnegie Mellon University
#
# SPDX-License-Identifier: Apache-2.0

# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger

setup_logger()

import copy
import glob
import json
import os
import pickle as pk
import shutil
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
from detectron2.data.datasets import load_coco_json, register_coco_instances
from detectron2.data.samplers import InferenceSampler, TrainingSampler
from detectron2.evaluation import COCOEvaluator
from detectron2.modeling import build_model
from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputs, fast_rcnn_inference
from detectron2.structures.image_list import ImageList
from detectron2.utils.events import EventStorage
from detectron2.utils.visualizer import ColorMode, Visualizer
from sklearn.svm import SVC
from tqdm import tqdm


class AutoDetector:
    def __init__(
        self,
        target_name,
        thres=0.5,
        device="cuda",
        feature_extractor="FRCNN_FBNet",
    ):
        self.target_name = target_name
        self.device = device
        self.task_dir = os.path.join("./autoDet_tasks/", self.target_name)
        os.makedirs(self.task_dir, exist_ok=True)
        self.labeled_img_dir = os.path.join(self.task_dir, "labeled")
        os.makedirs(self.labeled_img_dir, exist_ok=True)
        self.manual_anno_dir = os.path.join(self.task_dir, "manual_anno")
        os.makedirs(self.manual_anno_dir, exist_ok=True)
        self.svc_model_dir = os.path.join(self.task_dir, "svc_models_bus")
        os.makedirs(self.svc_model_dir, exist_ok=True)
        self.svc_cache_dir = os.path.join(self.task_dir, "svc_cache_bus")
        os.makedirs(self.svc_cache_dir, exist_ok=True)

        self.device = device
        self.confidence_thres = thres
        self.feature_extractor = feature_extractor

        self.cfg = self._get_cfg()
        self.model = self._get_model()
        self.model.eval()

        MetadataCatalog.get("empty_dataset").thing_classes = ["positive"]
        self.metadata = MetadataCatalog.get("empty_dataset")

        self.model_version = 0
        self.dataset_name = "svc_trainset"

        self._load_svm_model()

        MetadataCatalog.get("empty_dataset").thing_classes = ["positive"]
        self.metadata = MetadataCatalog.get("empty_dataset")

    def _load_svm_model(self):
        file_list = glob.glob(os.path.join(self.svc_model_dir, "*.pkl"))
        if len(file_list) == 0:
            self._svc = None
            self._svc_cached_trainset = None
        else:
            file_list.sort(key=os.path.getmtime)
            print("Loaded SVM model from ", file_list[-1])
            with open(file_list[-1], "rb") as file:
                self._svc = pk.load(file)
            self.model_version = int(file_list[-1].split("_")[-1][:-4])

            cache_list = glob.glob(os.path.join(self.svc_cache_dir, "*.pkl"))
            cache_list.sort(key=os.path.getmtime)
            print("Loaded SVM cached training set from ", cache_list[-1])
            with open(cache_list[-1], "rb") as file:
                self._svc_cached_trainset = pk.load(file)

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

    def _get_data_augmentations(self, deluxe=True):
        if deluxe:
            augs = [
                T.RandomCrop("relative", (0.9, 0.9)),
                T.RandomBrightness(0.9, 1.1),
                T.RandomContrast(0.9, 1.1),
                T.ResizeShortestEdge(
                    [self.cfg.INPUT.MIN_SIZE_TEST, self.cfg.INPUT.MIN_SIZE_TEST],
                    self.cfg.INPUT.MAX_SIZE_TEST,
                ),
            ]
        else:
            augs = [
                T.ResizeShortestEdge(
                    [self.cfg.INPUT.MIN_SIZE_TEST, self.cfg.INPUT.MIN_SIZE_TEST],
                    self.cfg.INPUT.MAX_SIZE_TEST,
                )
            ]
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
            cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
                "faster_rcnn_fbnetv3a_C4.yaml"
            )  # Let training initialize from model zoo
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
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.confidence_thres
        cfg.MODEL.DEVICE = self.device

        # Following FSCE to provide more frontground proposals
        cfg.MODEL.RPN.POST_NMS_TOPK_TRAIN = 2000
        cfg.MODEL.RPN.PRE_NMS_TOPK_TEST = (
            4000  # it is actually for training on the server
        )
        cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 2000
        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 96
        #         cfg.MODEL.ROI_HEADS.IOU_THRESHOLD = 0.4
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

    def extract_box_features(self, batched_inputs, train=False):
        """
        Args:
            batched_inputs (list): a list that contains input to the model, the format of the inputs should follow
                                   https://detectron2.readthedocs.io/en/latest/tutorials/data_loading.html
        """
        # forward
        # Normalize, pad and batch the input images. (Preprocess_image)
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [self._get_normalizer()(x) for x in images]
        images = ImageList.from_tensors(images, self.model.backbone.size_divisibility)
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

    def _concatenate_annotations(self, new_anno_dict, full_anno_dict):
        full_anno_img_len = len(full_anno_dict["images"])
        assert full_anno_img_len == full_anno_dict["images"][-1]["id"]
        full_anno_annotations_len = len(full_anno_dict["annotations"])
        assert full_anno_annotations_len == full_anno_dict["annotations"][-1]["id"]

        for i, new_annotation in enumerate(new_anno_dict["annotations"]):
            new_annotation["id"] = i + 1 + full_anno_annotations_len

        for i, new_img in enumerate(new_anno_dict["images"]):
            old_img_id = new_img["id"]
            new_img_id = i + 1 + full_anno_img_len
            new_img["id"] = new_img_id
            for new_annotation in new_anno_dict["annotations"]:
                if new_annotation["image_id"] == old_img_id:
                    new_annotation["image_id"] = new_img_id

        full_anno_dict["images"].extend(new_anno_dict["images"])
        full_anno_dict["annotations"].extend(new_anno_dict["annotations"])
        return full_anno_dict

    def _expand_annotations(self, new_anno_filename):
        with open(
            os.path.join(self.manual_anno_dir, new_anno_filename), "r"
        ) as new_anno_file:
            new_anno_dict = json.load(new_anno_file)

        if not os.path.exists(
            os.path.join(self.manual_anno_dir, "full_annotations_bus.json")
        ):
            for i, new_annotation in enumerate(new_anno_dict["annotations"]):
                new_annotation["id"] = i + 1

            for i, new_img in enumerate(new_anno_dict["images"]):
                old_img_id = new_img["id"]
                new_img_id = i + 1
                new_img["id"] = new_img_id
                for new_annotation in new_anno_dict["annotations"]:
                    if new_annotation["image_id"] == old_img_id:
                        new_annotation["image_id"] = new_img_id
            with open(
                os.path.join(self.manual_anno_dir, "full_annotations_bus.json"), "w"
            ) as full_anno_file:
                json.dump(new_anno_dict, full_anno_file)
        else:
            with open(
                os.path.join(self.manual_anno_dir, "full_annotations_bus.json"), "r+"
            ) as full_anno_file:
                full_anno_dict = json.load(full_anno_file)
                full_anno_dict = self._concatenate_annotations(
                    new_anno_dict, full_anno_dict
                )
                full_anno_file.seek(0)
                json.dump(full_anno_dict, full_anno_file)
                full_anno_file.truncate()

        # os.remove(os.path.join(self.manual_anno_dir, new_anno_filename))
        return

    def _register_trainset(self, json_path, image_path):
        if self.dataset_name in DatasetCatalog.list():
            DatasetCatalog.remove(self.dataset_name)
        register_coco_instances(self.dataset_name, {}, json_path, image_path)

    def train_svc(self, new_anno_filename):
        # self._expand_annotations(new_anno_filename)
        # full_anno_filename = os.path.join(self.manual_anno_dir, "full_annotations_bus.json")

        ## cached feature vectors instead of traverse the entire training set
        full_anno_filename = os.path.join(
            self.manual_anno_dir, "new_annotations_bus.json"
        )
        shutil.copy2(
            os.path.join(self.manual_anno_dir, new_anno_filename), full_anno_filename
        )

        self._register_trainset(full_anno_filename, self.labeled_img_dir)

        data_len = len(DatasetCatalog.get(self.dataset_name))
        print("data_len = ", data_len)
        batch_size = data_len
        data_loader = build_detection_train_loader(
            DatasetCatalog.get(self.dataset_name),
            mapper=DatasetMapper(
                self.cfg, is_train=True, augmentations=self._get_data_augmentations()
            ),
            sampler=TrainingSampler(data_len, shuffle=False),
            total_batch_size=batch_size,
        )
        data_loader_it = iter(data_loader)
        iter_num = 10

        feature_vec_list = []
        proposals_with_gt = []
        svm_start_time1 = time.time()
        with EventStorage() as storage:  # this is needed by function label_and_sample_proposals
            with torch.no_grad():
                for idx in range(iter_num):
                    batched_inputs = next(data_loader_it)
                    # print([d['instances'].gt_classes for d in batched_inputs])

                    box_features, proposals = self.extract_box_features(
                        batched_inputs, train=True
                    )
                    # print([p.gt_classes for p in proposals])

                    # For SVM training: X and y
                    feature_vec_list.extend(box_features.squeeze())
                    proposals_with_gt.extend(proposals)

        # print(len(feature_vec_list))
        # print(len(proposals_with_gt))
        X = torch.vstack(feature_vec_list).cpu().detach().numpy()
        y = (
            torch.vstack([p.gt_classes.reshape((-1, 1)) for p in proposals_with_gt])
            .cpu()
            .detach()
            .numpy()
            .ravel()
        )
        print("len_y = ", y.shape[0])
        if not self._svc_cached_trainset:
            self._svc_cached_trainset = {"X": X, "y": y}
        else:
            self._svc_cached_trainset["X"] = np.concatenate(
                (self._svc_cached_trainset["X"], X), axis=0
            )
            self._svc_cached_trainset["y"] = np.concatenate(
                (self._svc_cached_trainset["y"], y), axis=0
            )

        with open(
            os.path.join(
                self.svc_cache_dir, "for_model_{}.pkl".format(self.model_version + 1)
            ),
            "wb",
        ) as f:
            pk.dump(self._svc_cached_trainset, f)

        X = copy.deepcopy(self._svc_cached_trainset["X"])
        y = copy.deepcopy(self._svc_cached_trainset["y"])
        rng = np.random.default_rng()

        if self.cfg.MODEL.ROI_HEADS.NUM_CLASSES == 1:
            pos_num = y[y == 0].shape[0]
            neg_num = y[y == 1].shape[0]
            print("positive : negative = ", pos_num, " : ", neg_num)
            svm_start_time2 = time.time()

            radio = 4
            selected_neg_num = min(pos_num * radio, neg_num)
            random_numbers = rng.choice(neg_num, size=selected_neg_num, replace=False)

            pos_X = X[y == 0, :]
            neg_X = X[y == 1, :][random_numbers]
            pos_y = y[y == 0]
            neg_y = y[y == 1][random_numbers]

            X = np.concatenate((pos_X, neg_X), axis=0)
            y = np.concatenate((pos_y, neg_y), axis=0)
            print("X shape = ", X.shape)
            print("y shape = ", y.shape)

            clf = SVC(
                random_state=42, probability=True, class_weight="balanced", kernel="rbf"
            )
            # clf = SVC(random_state=42, probability=True, class_weight={0:radio, 1: 1}, kernel='linear')
            clf.fit(X, y)
            svm_train_time = time.time() - svm_start_time2
        else:
            pos_num = y[y == 0].shape[0]
            hard_neg_num = y[y == 1].shape[0]
            common_neg_num = y[y == 2].shape[0]
            print(
                "positive : hard negative : common negative = ",
                pos_num,
                " : ",
                hard_neg_num,
                " : ",
                common_neg_num,
            )
            svm_start_time2 = time.time()

            radio = 4
            selected_neg_num = min(pos_num * radio - hard_neg_num, common_neg_num)
            random_numbers = rng.choice(
                common_neg_num, size=selected_neg_num, replace=False
            )

            pos_X = X[y == 0, :]
            hard_neg_X = X[y == 1, :]
            comm_neg_X = X[y == 2, :][random_numbers]
            pos_y = y[y == 0]
            hard_neg_y = y[y == 1]
            comm_neg_y = y[y == 2][random_numbers]

            X = np.concatenate((pos_X, hard_neg_X, comm_neg_X), axis=0)
            y = np.concatenate((pos_y, hard_neg_y, comm_neg_y), axis=0)
            print("X shape = ", X.shape)
            print("y shape = ", y.shape)

            clf = SVC(
                random_state=42,
                probability=True,
                class_weight="balanced",
                kernel="rbf",
                decision_function_shape="ovo",
            )
            clf.fit(X, y)
            svm_train_time = time.time() - svm_start_time2

        print("SVM retraining time: {} s".format(svm_train_time))
        print(
            "SVM retraining time including inference: {} s".format(
                time.time() - svm_start_time1
            )
        )
        self._svc = clf

        self._notify_and_save_model(clf)

    def _notify_and_save_model(self, svc):
        self.model_version += 1
        self._save_model(svc)
        flag_file = os.path.join(self.svc_model_dir, "update_flag")
        with open(flag_file, "w") as f:
            f.write(str(self.model_version))
        print(
            "SVM model gets updated to version {}, notified busEdge and saved the model as file".format(
                self.model_version
            )
        )

    def _save_model(self, svc):
        output_model_path = os.path.join(
            self.svc_model_dir, "svc_model_{}.pkl".format(self.model_version)
        )
        with open(output_model_path, "wb") as file:
            pk.dump(svc, file)
