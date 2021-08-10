# SPDX-FileCopyrightText: 2021 Carnegie Mellon University
#
# SPDX-License-Identifier: Apache-2.0

import logging

import cv2
import imageio
import numpy as np
from gabriel_protocol import gabriel_pb2
from gabriel_server import cognitive_engine
from protocol import busedge_pb2
from utils.exif import set_gps_location

logger = logging.getLogger(__name__)


class ExampleEngine(cognitive_engine.Engine):
    def __init__(self, source_name):
        # -----------------------------------------------------------#
        # Initialize your model here
        # -----------------------------------------------------------#

        logger.info("Cognitive Engine has been initilized")

        self.source_name = source_name

    def handle(self, input_frame):
        # -----------------------------------------------------------#
        # read input from gabriel server
        # -----------------------------------------------------------#
        if input_frame.payload_type != gabriel_pb2.PayloadType.IMAGE:
            status = gabriel_pb2.ResultWrapper.Status.WRONG_INPUT_FORMAT
            return cognitive_engine.create_result_wrapper(status)
        extras = cognitive_engine.unpack_extras(navlab_pb2.EngineFields, input_frame)
        gps = [
            extras.gps_data.latitude,
            extras.gps_data.longitude,
            extras.gps_data.altitude,
        ]
        img_name = extras.image_filename
        img_array = np.asarray(bytearray(input_frame.payloads[0]), dtype=np.int8)
        img = cv2.imdecode(img_array, -1)  # cv::IMREAD_UNCHANGED = -1

        # -----------------------------------------------------------#
        # Run detector on client input
        # -----------------------------------------------------------#
        # result = your_model(img)

        result_wrapper = gabriel_pb2.ResultWrapper()
        result_wrapper.result_producer_name.value = self.source_name
        result_wrapper.status = gabriel_pb2.ResultWrapper.Status.SUCCESS
        # -----------------------------------------------------------#
        # if you need to send the results back to the client
        # -----------------------------------------------------------#
        # result = gabriel_pb2.ResultWrapper.Result()
        # result.payload_type = gabriel_pb2.PayloadType.IMAGE
        # result.payload = input_frame.payloads[0]
        # result_wrapper.results.append(result)

        return result_wrapper
