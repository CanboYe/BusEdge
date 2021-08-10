# SPDX-FileCopyrightText: 2021 Carnegie Mellon University
#
# SPDX-License-Identifier: Apache-2.0

import logging
import os
import pickle as pk
import time

import cv2
from gabriel_protocol import gabriel_pb2

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def consumer(result_wrapper):
    # logger.info('Consuming the results from server')
    if len(result_wrapper.results) > 0:
        result = result_wrapper.results[0]
        if result.payload_type == gabriel_pb2.PayloadType.OTHER:
            model_bytes = result.payload
            source_name = result_wrapper.result_producer_name.value
            notify_and_save_model(model_bytes, source_name)
        else:
            pass


def notify_and_save_model(model_bytes, source_name):
    save_dir = "./nodes/autodet_node/models/" + str(source_name)
    os.makedirs(save_dir, exist_ok=True)
    with open(save_dir + "/svc_model.pkl", "wb") as model_file:
        model_file.write(model_bytes)
    with open(save_dir + "/update_flag", "w") as f:
        f.write("1")
    logger.info("SVM model gets updated, notified rosnode and saved the model as file")
