# SPDX-FileCopyrightText: 2021 Carnegie Mellon University
#
# SPDX-License-Identifier: Apache-2.0

import logging
import sys

import cv2
import imageio
import numpy as np
from busedge_protocol import busedge_pb2

# from gabriel_server import cognitive_engine
from gabriel_protocol import gabriel_pb2
from gabriel_server import cognitive_engine

logger = logging.getLogger(__name__)


class NoopEngine(cognitive_engine.Engine):
    def __init__(self, source_name):
        logger.info("Noop has been initilized")
        self.source_name = source_name

    def handle(self, input_frame):
        if input_frame.payload_type != gabriel_pb2.PayloadType.IMAGE:
            status = gabriel_pb2.ResultWrapper.Status.WRONG_INPUT_FORMAT
            return cognitive_engine.create_result_wrapper(status)
        extras = cognitive_engine.unpack_extras(busedge_pb2.EngineFields, input_frame)

        # -----------------------------------------------------------#
        # No operation
        img_array = np.asarray(bytearray(input_frame.payloads[0]), dtype=np.int8)
        img = cv2.imdecode(img_array, -1)
        show_flag = True
        if show_flag:
            cv2.namedWindow("noop engine", 0)
            cv2.imshow("noop engine", img[:, :, ::-1])
            cv2.waitKey(1)
        logger.info("Noop engine finished one iteration.")
        # -----------------------------------------------------------#

        result_wrapper = gabriel_pb2.ResultWrapper()
        result_wrapper.result_producer_name.value = self.source_name
        result_wrapper.status = gabriel_pb2.ResultWrapper.Status.SUCCESS

        return result_wrapper
