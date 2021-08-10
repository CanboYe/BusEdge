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

# import some common detectron2 utilities
from detectron2 import model_zoo
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
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.modeling import build_model
from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputs, fast_rcnn_inference
from detectron2.structures.image_list import ImageList
from detectron2.utils.events import EventStorage
from detectron2.utils.visualizer import ColorMode, Visualizer
from sklearn.svm import SVC
from tqdm import tqdm

# from fast_rcnn import FastRCNNOutputs


class AutoDetector:
    def __init__(
        self,
        target_name,
        thres=0.7,
        device="cuda",
        num_cls=2,
        feature_extractor="FRCNN_R101_FPN",
        use_svm=False,
        svm_num_cls=1,
        max_svm_update=3,
    ):
        self.target_name = target_name
        self.device = device
        self.num_cls = num_cls
        self.use_svm = use_svm
        self.thres = thres
        self.svm_num_cls = svm_num_cls
        # according to experiments, 2_cls does not improve svm but increase training and inference time
        self.max_svm_update = max_svm_update

        self.task_dir = os.path.join("./autoDet_tasks/", self.target_name)
        os.makedirs(self.task_dir, exist_ok=True)
        self.labeled_img_dir = os.path.join(self.task_dir, "labeled")
        os.makedirs(self.labeled_img_dir, exist_ok=True)
        self.manual_anno_dir = os.path.join(self.task_dir, "manual_anno")
        os.makedirs(self.manual_anno_dir, exist_ok=True)
        self.svc_model_dir = os.path.join(self.task_dir, "svc_models_cloudlet")
        os.makedirs(self.svc_model_dir, exist_ok=True)
        self.frcnn_model_dir = os.path.join(self.task_dir, "frcnn_models")
        os.makedirs(self.frcnn_model_dir, exist_ok=True)
        self.svc_cache_dir = os.path.join(self.task_dir, "svc_cache_cloudlet")
        os.makedirs(self.svc_cache_dir, exist_ok=True)

        self.last_max_iter = 0

        self.cfg = get_cfg()
        self.feature_extractor = feature_extractor
        self.model = self._get_model()
        self.model.eval()

        self.model_version = 0
        self._load_model()
        self.dataset_name = "svc_trainset"

        MetadataCatalog.get("empty_dataset").thing_classes = [
            "positive",
            "hard negative",
        ]
        self.metadata = MetadataCatalog.get("empty_dataset")

    def _load_model(self):
        file_list_frcnn = glob.glob(os.path.join(self.frcnn_model_dir, "model_v_*.pth"))
        if len(file_list_frcnn) != 0:
            file_list_frcnn.sort(key=os.path.getmtime)
            newest_version = int(file_list_frcnn[-1].split("/")[-1].split("_")[-1][:-4])
            self.update_model(file_list_frcnn[-1])
            self.model_version = newest_version
            print(
                "FRCNN model has been updated to version {}!".format(self.model_version)
            )
        else:
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

    def _get_data_augmentations(self):
        augs = [
            T.RandomCrop("relative", (0.9, 0.9)),
            # T.RandomFlip(prob=0.5),
            T.RandomBrightness(0.9, 1.1),
            T.RandomContrast(0.9, 1.1),
            T.ResizeShortestEdge(
                [self.cfg.INPUT.MIN_SIZE_TEST, self.cfg.INPUT.MIN_SIZE_TEST],
                self.cfg.INPUT.MAX_SIZE_TEST,
            ),
        ]
        #         augs = [T.ResizeShortestEdge(
        #                     [self.cfg.INPUT.MIN_SIZE_TEST, self.cfg.INPUT.MIN_SIZE_TEST],
        #                     self.cfg.INPUT.MAX_SIZE_TEST,
        #                 )]
        return augs

    def _get_model(self):
        if self.feature_extractor == "FRCNN_R50_FPN":
            self.cfg.merge_from_file(
                model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
            )
            self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
                "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
            )

        elif self.feature_extractor == "FRCNN_R101_FPN":
            self.cfg.merge_from_file(
                model_zoo.get_config_file(
                    "COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"
                )
            )
            self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
                "COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"
            )
        else:
            raise NotImplementedError(
                "unknown feature extractor " + self.feature_extractor
            )

        self.cfg.MODEL.DEVICE = self.device
        self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = self.num_cls
        self.cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
        self.cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION = 0.95
        self.cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.3
        self.cfg.TEST.DETECTIONS_PER_IMAGE = 20
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.thres

        self.cfg.MODEL.RPN.PRE_NMS_TOPK_TRAIN = 6000
        self.cfg.MODEL.RPN.POST_NMS_TOPK_TRAIN = 2000
        self.cfg.MODEL.RPN.PRE_NMS_TOPK_TEST = 3000
        self.cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 1000
        #         self.cfg.OUTPUT_DIR = self.output_dir
        #         os.makedirs(self.cfg.OUTPUT_DIR, exist_ok=True)

        model = build_model(self.cfg)
        checkpointer = DetectionCheckpointer(model)
        checkpointer.load(self.cfg.MODEL.WEIGHTS)
        return model

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
        # print('features shape:', features['p3'].shape)
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
            os.path.join(self.manual_anno_dir, "full_annotations_cloudlet.json")
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
                os.path.join(self.manual_anno_dir, "full_annotations_cloudlet.json"),
                "w",
            ) as full_anno_file:
                json.dump(new_anno_dict, full_anno_file)
        else:
            with open(
                os.path.join(self.manual_anno_dir, "full_annotations_cloudlet.json"),
                "r+",
            ) as full_anno_file:
                full_anno_dict = json.load(full_anno_file)
                full_anno_dict = self._concatenate_annotations(
                    new_anno_dict, full_anno_dict
                )
                full_anno_file.seek(0)
                json.dump(full_anno_dict, full_anno_file)
                full_anno_file.truncate()
        return

    def _register_trainset(self, json_path, image_path):
        if self.dataset_name in DatasetCatalog.list():
            DatasetCatalog.remove(self.dataset_name)
            MetadataCatalog.remove(self.dataset_name)
        register_coco_instances(self.dataset_name, {}, json_path, image_path)

    def retrain_svm(self, new_anno_filename):
        self._expand_annotations(new_anno_filename)
        # full_anno_filename = os.path.join(
        #     self.manual_anno_dir, "full_annotations_cloudlet.json"
        # )

        ## cached feature vectors instead of traverse the entire training set
        full_anno_filename = os.path.join(
            self.manual_anno_dir, "new_annotations_cloudlet.json"
        )
        shutil.copy2(
            os.path.join(self.manual_anno_dir, new_anno_filename), full_anno_filename
        )

        self._register_trainset(full_anno_filename, self.labeled_img_dir)

        data_len = len(DatasetCatalog.get(self.dataset_name))
        print("data_len = ", data_len)
        svm_start_time1 = time.time()
        batch_size = data_len
        data_loader = build_detection_train_loader(
            DatasetCatalog.get(self.dataset_name),
            mapper=DatasetMapper(
                self.cfg, is_train=True, augmentations=self._get_data_augmentations()
            ),
            sampler=TrainingSampler(data_len, shuffle=True),
            total_batch_size=batch_size,
        )
        data_loader_it = iter(data_loader)
        iter_num = 5

        feature_vec_list = []
        proposals_with_gt = []
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
                    feature_vec_list.extend(box_features)
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

        if self.svm_num_cls == 1:
            pos_num = y[y == 0].shape[0]
            neg_num = y[y == self.num_cls].shape[0]
            print("positive : negative = ", pos_num, " : ", neg_num)
            svm_start_time2 = time.time()

            radio = 4
            selected_neg_num = min(pos_num * radio, neg_num)
            random_numbers = rng.choice(neg_num, size=selected_neg_num, replace=False)

            pos_X = X[y == 0, :]
            neg_X = X[y == self.num_cls, :][random_numbers]
            pos_y = y[y == 0]
            neg_y = y[y == self.num_cls][random_numbers]

            X = np.concatenate((pos_X, neg_X), axis=0)
            y = np.concatenate((pos_y, neg_y), axis=0)
            print("X shape = ", X.shape)
            print("y shape = ", y.shape)

            clf = SVC(
                random_state=42, probability=True, class_weight="balanced", kernel="rbf"
            )
            # clf = SVC(random_state=42, probability=True, class_weight={0:radio, 1: 1}, kernel='linear')
            clf.fit(X, y)
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

            radio = 2
            selected_neg_num = min(pos_num * radio + hard_neg_num, common_neg_num)
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

    def _split_anno(self, step_num, step_size=10):
        anno_json_list = []
        full_anno_filename = os.path.join(
            self.manual_anno_dir, "full_annotations_cloudlet.json"
        )
        with open(full_anno_filename, "r") as full_anno_file:
            full_anno_dict = json.load(full_anno_file)

        # print(len(full_anno_dict['images']))
        # print(len(full_anno_dict['annotations']))

        coco_anno_dict = {
            "info": [],
            "licenses": [],
            "categories": [
                {"id": 1, "name": "positive", "supercategory": ""},
                {"id": 2, "name": "hard negative", "supercategory": ""},
            ],
            "images": [],
            "annotations": [],
        }
        if self.cfg.MODEL.ROI_HEADS.NUM_CLASSES == 1:
            coco_anno_dict["categories"] = [
                {"id": 1, "name": "positive", "supercategory": ""}
            ]
        for step in range(step_num):
            image_list = full_anno_dict["images"][
                step * step_size : (step + 1) * step_size
            ]
            image_ids = [img_dic["id"] for img_dic in image_list]
            anno_list = []
            for anno in full_anno_dict["annotations"]:
                if self.cfg.MODEL.ROI_HEADS.NUM_CLASSES == 1:
                    if anno["category_id"] == 2:
                        continue
                img_id = anno["image_id"]
                if img_id in set(image_ids):
                    anno_list.append(anno)
            coco_anno_dict["images"].extend(image_list)
            coco_anno_dict["annotations"].extend(anno_list)
            if self.cfg.MODEL.ROI_HEADS.NUM_CLASSES == 1:
                tmp_dir = "./manual_anno_split_pos"
            else:
                tmp_dir = "./manual_anno_split"
            os.makedirs(tmp_dir, exist_ok=True)
            json_file_name = tmp_dir + "/annotations_{}.json".format(step + 1)
            anno_json_list.append(json_file_name)
            with open(json_file_name, "w") as json_file:
                json.dump(coco_anno_dict, json_file)
        #             print(len(coco_anno_dict['images']))
        #             print(len(coco_anno_dict['annotations']))
        # coco_anno_dict['images'] = []
        # coco_anno_dict['annotations'] = []
        return anno_json_list

    def retrain_finetune(self, new_anno_filename):

        self._expand_annotations(new_anno_filename)
        full_anno_filename = os.path.join(
            self.manual_anno_dir, "full_annotations_cloudlet.json"
        )
        self._register_trainset(full_anno_filename, self.labeled_img_dir)

        data_len = len(DatasetCatalog.get(self.dataset_name))
        print("data_len = ", data_len)

        self.cfg.DATASETS.TRAIN = (self.dataset_name,)
        self.cfg.DATASETS.TEST = ()
        self.cfg.DATALOADER.NUM_WORKERS = 2
        self.cfg.SOLVER.IMS_PER_BATCH = 2
        self.cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
        # self.cfg.SOLVER.MAX_ITER = 50 * data_len
        self.cfg.SOLVER.STEPS = []  # do not decay learning rate
        self.cfg.OUTPUT_DIR = self.frcnn_model_dir

        # self.cfg.SOLVER.MAX_ITER = self.last_max_iter + 800
        self.cfg.SOLVER.MAX_ITER = max(data_len * 40, self.last_max_iter + 600)
        self.last_max_iter = self.cfg.SOLVER.MAX_ITER
        # os.makedirs(self.cfg.OUTPUT_DIR, exist_ok=True)
        trainer = DefaultTrainer(self.cfg)
        trainer.resume_or_load(resume=True)
        for p in trainer.model.backbone.parameters():
            p.requires_grad = False
        print("froze backbone parameters")

        if self.model_version < 3:
            for p in trainer.model.proposal_generator.parameters():
                p.requires_grad = False
            print("froze proposal_generator parameters")

            for p in trainer.model.roi_heads.box_head.parameters():
                p.requires_grad = False
            print("froze box_head parameters")

        trainer.train()

        self.model_version += 1
        model_path = os.path.join(self.cfg.OUTPUT_DIR, "model_final.pth")
        self.update_model(new_model_path=model_path)
        copy_model_path = os.path.join(
            self.cfg.OUTPUT_DIR, "model_v_{}.pth".format(self.model_version)
        )
        shutil.copy2(model_path, copy_model_path)

        flag_file = os.path.join(self.frcnn_model_dir, "update_flag")
        with open(flag_file, "w") as f:
            f.write(str(self.model_version))
        print(
            "FRCNN model gets updated to version {}, notified busEdge and saved the model as file".format(
                self.model_version
            )
        )

    def retrain_finetune_svm(self, new_anno_filename):
        if self.model_version < self.max_svm_update:
            self.retrain_svm(new_anno_filename)
        else:
            print("Now start to do fine-tuning")
            self.retrain_finetune(new_anno_filename)

    def update_model(self, new_model_path="output/model_final.pth"):
        checkpointer = DetectionCheckpointer(self.model)
        checkpointer.load(new_model_path)
        self.model.eval()
        print("Loaded FRCNN model from ", new_model_path)

    def check_model_update(self):
        file_list_frcnn = glob.glob(os.path.join(self.frcnn_model_dir, "model_v_*.pth"))

        if self.use_svm:
            if self.model_version <= self.max_svm_update and len(file_list_frcnn) == 0:
                file_list = glob.glob(os.path.join(self.svc_model_dir, "*.pkl"))
                if len(file_list) == 0:
                    return False
                else:
                    file_list.sort(key=os.path.getmtime)
                    newest_version = int(
                        file_list[-1].split("/")[-1].split("_")[-1][:-4]
                    )
                    if self.model_version < newest_version:
                        print("Loaded SVM model from ", file_list[-1])
                        with open(file_list[-1], "rb") as file:
                            self._svc = pk.load(file)
                        self.model_version = newest_version
                        print(
                            "SVM model has been updated to version {}!".format(
                                self.model_version
                            )
                        )
                    return True

        if len(file_list_frcnn) == 0:
            return False
        else:
            file_list_frcnn.sort(key=os.path.getmtime)
            newest_version = int(file_list_frcnn[-1].split("/")[-1].split("_")[-1][:-4])
            if self.model_version < newest_version:
                self.update_model(file_list_frcnn[-1])
                self.model_version = newest_version
                print(
                    "FRCNN model has been updated to version {}!".format(
                        self.model_version
                    )
                )
            return True

    def predict(self, im, save_img_dir=None):
        """
        Args:
            im (np.array): a image in RGB format
        """
        if not self.check_model_update():
            assert self.model_version == 0
            print("No model exists, bootstrapping is needed.")
            return np.array([[]]), np.array([])

        height, width = im.shape[:2]
        augs = T.ResizeShortestEdge(
            [self.cfg.INPUT.MIN_SIZE_TEST, self.cfg.INPUT.MIN_SIZE_TEST],
            self.cfg.INPUT.MAX_SIZE_TEST,
        )
        image = augs.get_transform(im).apply_image(im)
        image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
        batched_inputs = [{"height": height, "width": width, "image": image}]

        with torch.no_grad():
            box_features, proposals = self.extract_box_features(
                batched_inputs, train=False
            )
            (
                pred_class_logits,
                pred_proposal_deltas,
            ) = self.model.roi_heads.box_predictor(box_features)
            #             print('pred_proposal_deltas', pred_proposal_deltas.shape) # [1000, 8]
            #             print('pred_class_logits', pred_class_logits.shape) # [1000, 3]

            if self.use_svm and self.model_version <= self.max_svm_update:
                # SVM
                X = box_features.to("cpu").detach().numpy()
                pred_class_logits = self._svc.predict_log_proba(X)
                pred_class_logits = torch.from_numpy(pred_class_logits).to(self.device)
                # to fix bug: dets should have the same type as scores
                pred_class_logits = pred_class_logits.to(dtype=torch.float)
                #                 print('pred_class_logits_svm', pred_class_logits.shape)
                delta_num = 4 * self.svm_num_cls
            else:
                delta_num = 4 * self.num_cls
            predictions = pred_class_logits, pred_proposal_deltas[:, :delta_num]
            pred_instances, _ = self.model.roi_heads.box_predictor.inference(
                predictions, proposals
            )

        processed_results = []
        for results_per_image in pred_instances:
            r = detector_postprocess(results_per_image, height, width)
            processed_results.append({"instances": r})

        output = processed_results[0]["instances"].to("cpu")
        #         print(output)
        boxes = output.pred_boxes.tensor.numpy()
        scores = output.scores.numpy()
        pred_classes = output.pred_classes.numpy()
        # boxes: y_lt, x_lt, y_rb, x_rb
        #         if scores.shape[0] > 0:
        #             print(
        #                 "\n\nSVM postprocessed instance for image 0:\n",
        #                 processed_results[0],
        #                 "\n",
        #             )

        boxes, scores = self._filter_cls_and_small(
            boxes, scores, pred_classes, thres_area=1000
        )
        # print(boxes.shape)
        # print(scores.shape)
        if scores.shape[0] > 0:
            self._visualize_results(im, processed_results[0]["instances"], save_img_dir)
        return boxes, scores

    def predict_livemap(self, im):
        """
        Args:
            im (np.array): a image in RGB format
        """
        if not self.check_model_update():
            assert self.model_version == 0
            print("No model exists, bootstrapping is needed.")
            return np.array([[]]), np.array([])

        height, width = im.shape[:2]
        augs = T.ResizeShortestEdge(
            [self.cfg.INPUT.MIN_SIZE_TEST, self.cfg.INPUT.MIN_SIZE_TEST],
            self.cfg.INPUT.MAX_SIZE_TEST,
        )
        image = augs.get_transform(im).apply_image(im)
        image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
        batched_inputs = [{"height": height, "width": width, "image": image}]

        with torch.no_grad():
            box_features, proposals = self.extract_box_features(
                batched_inputs, train=False
            )
            (
                pred_class_logits,
                pred_proposal_deltas,
            ) = self.model.roi_heads.box_predictor(box_features)
            #             print('pred_proposal_deltas', pred_proposal_deltas.shape) # [1000, 8]
            #             print('pred_class_logits', pred_class_logits.shape) # [1000, 3]

            if self.use_svm and self.model_version <= self.max_svm_update:
                # SVM
                X = box_features.to("cpu").detach().numpy()
                pred_class_logits = self._svc.predict_log_proba(X)
                pred_class_logits = torch.from_numpy(pred_class_logits).to(self.device)
                # to fix bug: dets should have the same type as scores
                pred_class_logits = pred_class_logits.to(dtype=torch.float)
                #                 print('pred_class_logits_svm', pred_class_logits.shape)
                delta_num = 4 * self.svm_num_cls
            else:
                delta_num = 4 * self.num_cls
            predictions = pred_class_logits, pred_proposal_deltas[:, :delta_num]
            pred_instances, _ = self.model.roi_heads.box_predictor.inference(
                predictions, proposals
            )

        processed_results = []
        for results_per_image in pred_instances:
            r = detector_postprocess(results_per_image, height, width)
            processed_results.append({"instances": r})

        outputs = processed_results[0]["instances"]
        return outputs

    def _visualize_results(self, raw_image, result_show, save_img_dir):
        v = Visualizer(
            raw_image, metadata=self.metadata, scale=1.0, instance_mode=ColorMode.IMAGE
        )
        v = v.draw_instance_predictions(
            result_show[result_show.pred_classes == 0].to("cpu")
        )
        if save_img_dir is not None:
            os.makedirs(os.path.dirname(save_img_dir), exist_ok=True)
            cv2.imwrite(save_img_dir, v.get_image()[:, :, ::-1])
        else:
            window_name = "Cloudlet Results"
            cv2.namedWindow(
                window_name,
                cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_NORMAL,
            )

            width, height = 480, 270
            cv2.moveWindow(window_name, 2000, 80 + (height + 30) * 2)
            cv2.resizeWindow(window_name, width, height)
            cv2.imshow(window_name, v.get_image()[:, :, ::-1])
            cv2.waitKey(1)

    def _filter_cls_and_small(self, boxes, scores, pred_classes, thres_area=1000):
        filtered_boxes = []
        filtered_scores = []
        results_len = scores.shape[0]

        for i in range(results_len):
            box = boxes[i]
            score = scores[i]
            cls = pred_classes[i]
            if cls == 0:
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
