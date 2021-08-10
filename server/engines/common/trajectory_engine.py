# SPDX-FileCopyrightText: 2021 Carnegie Mellon University
#
# SPDX-License-Identifier: Apache-2.0

import logging
import sys
import time

import cv2
import imageio
import numpy as np
from busedge_protocol import busedge_pb2
from gabriel_protocol import gabriel_pb2
from gabriel_server import cognitive_engine

# libs for database
from utils.db_utils import DB_Manager

logger = logging.getLogger(__name__)


class TrajectoryEngine(cognitive_engine.Engine):
    def __init__(self, source_name):
        self.source_name = source_name
        logger.info("Trajectory Engine has been initilized")
        self.db_manager = DB_Manager()

    def handle(self, input_frame):
        if input_frame.payload_type != gabriel_pb2.PayloadType.OTHER:
            status = gabriel_pb2.ResultWrapper.Status.WRONG_INPUT_FORMAT
            return cognitive_engine.create_result_wrapper(status)
        extras = cognitive_engine.unpack_extras(busedge_pb2.EngineFields, input_frame)

        # -----------------------------------------------------------#
        self.db_manager.insert_trajectory(
            extras.gps_data.latitude,
            extras.gps_data.longitude,
            extras.gps_data.altitude,
        )

        logger.info(
            "Inserted GPS: ({:.4f},{:.4f},{:.4f})".format(
                extras.gps_data.latitude,
                extras.gps_data.longitude,
                extras.gps_data.altitude,
            )
        )

        result_wrapper = gabriel_pb2.ResultWrapper()
        result_wrapper.result_producer_name.value = self.source_name
        result_wrapper.status = gabriel_pb2.ResultWrapper.Status.SUCCESS

        return result_wrapper
