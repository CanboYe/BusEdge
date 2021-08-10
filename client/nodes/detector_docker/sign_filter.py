# SPDX-FileCopyrightText: 2021 Carnegie Mellon University
#
# SPDX-License-Identifier: Apache-2.0

# Inspired by https://github.com/tensorflow/models/blob/master/research/object_detection/colab_tutorials/object_detection_tutorial.ipynb
# SPDX-FileCopyrightText: 2020 TensorFlow Authors

import time
from pathlib import Path

import cv2
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from object_detection.utils import visualization_utils as vis_util
from object_detection.utils.ops import normalized_to_image_coordinates

category_index = {1: "sign"}


class SignFilter:
    def __init__(self, model_dir):
        tf.config.threading.set_inter_op_parallelism_threads(1)
        tf.config.threading.set_intra_op_parallelism_threads(1)
        tf.compat.v1.enable_eager_execution()
        self.model = tf.compat.v2.saved_model.load(str(model_dir))
        self.model_fn = self.model.signatures["serving_default"]

    def detect(self, image, min_score_thresh=None):
        image = np.asarray(image)
        input_tensor = tf.convert_to_tensor(image)[
            tf.newaxis, ...
        ]  # Convert to batched tensor
        output_dict = self.model_fn(input_tensor)

        # Convert to a usable form and remove batch dimension
        num_detections = int(output_dict.pop("num_detections").numpy()[0])
        output_dict["detection_boxes"] = normalized_to_image_coordinates(
            output_dict["detection_boxes"], np.array(image.shape)[[2, 0, 1]]
        )  # Use image coordinates
        output_dict = {
            key: value[0, :].numpy()  # TODO put back :num_detections to not get garbage
            for key, value in output_dict.items()
            if key in {"detection_boxes", "detection_classes", "detection_scores"}
        }

        if min_score_thresh is not None:
            confident_detections = output_dict["detection_scores"] > min_score_thresh
            output_dict = {k: v[confident_detections] for k, v in output_dict.items()}

        output_dict["num_detections"] = output_dict[
            "detection_boxes"
        ].size  # num_detections # TODO put back num_detections
        output_dict["detection_classes"] = output_dict["detection_classes"].astype(
            np.int64
        )

        return output_dict

    def display(self, image, output_dict):
        return vis_util.visualize_boxes_and_labels_on_image_array(
            image,
            output_dict["detection_boxes"],
            output_dict["detection_classes"],
            output_dict["detection_scores"],
            category_index,
            use_normalized_coordinates=False,
            line_thickness=4,
            agnostic_mode=True,
        )

    def send(self, image, min_score_thresh=0.6, show_flag=False):
        output_dict = self.detect(image, min_score_thresh)
        if output_dict["num_detections"] > 0:
            if show_flag:
                image_bbox = self.display(image, output_dict)
                cv2.namedWindow("Sign detector results", 0)
                cv2.imshow("Sign detector results", image_bbox[:, :, ::-1])
                cv2.waitKey(1)
            return True
        else:
            return False
