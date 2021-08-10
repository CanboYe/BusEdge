# SPDX-FileCopyrightText: 2021 Carnegie Mellon University
#
# SPDX-License-Identifier: Apache-2.0

import glob
import io
import os
import pickle as pk
import time

import cv2
import torch
from detector_fbnet import DelphiDetector


class DelphiFilter:
    def __init__(
        self, feature_extractor, svm_model_dir, svm_threshold=0.8, img_save_dir=None
    ):
        # torch.set_num_threads(8)
        self.feature_extractor = feature_extractor
        self.svm_threshold = svm_threshold
        self.svm_model_dir = svm_model_dir
        self.img_save_dir = img_save_dir
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = DelphiDetector(
            self.svm_threshold, self.device, self.feature_extractor, self.svm_model_dir
        )

    def detect_and_send(self, image, save_img_dir=None):
        tic = time.time()
        boxes, scores = self.model.predict_svc(image, save_img_dir)
        det_time = time.time() - tic
        print("detection takes time ", det_time)
        if scores.shape[0] == 0:
            return False, None
        results = {"boxes": boxes, "scores": scores}

        print("scores = ", scores)
        results_bytes_to_send = pk.dumps(results)
        return True, results_bytes_to_send

    def send(self, image):
        raise NotImplementedError

    def send_crop(self, image):
        raise NotImplementedError


if __name__ == "__main__":
    pass
