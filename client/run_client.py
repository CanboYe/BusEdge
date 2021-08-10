# SPDX-FileCopyrightText: 2021 Carnegie Mellon University
#
# SPDX-License-Identifier: Apache-2.0

import argparse
import logging
import multiprocessing
import os
import time

import cv2
from gabriel_client.push_source import Source
from gabriel_client.websocket_client import WebsocketClient
from nodes.common import docker_subsriber, image_noop_node, trajectory_node
from nodes.common.consumers import consumer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    server_addr = os.getenv("GABRIEL_SERVER_IP_ADDRESS")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-n",
        "--source-name",
        nargs="+",
        default=[],
        help="Set source name for this pipeline",
    )
    args = parser.parse_args()

    producer_wrappers = []

    ## Two common nodes: trajectory and no-operation nodes
    gps_noop_filter = Source("gps")
    p_gps = multiprocessing.Process(
        target=trajectory_node.run_node, args=(gps_noop_filter,)
    )
    p_gps.start()
    producer_wrappers.append(gps_noop_filter.get_producer_wrapper())

    # source_name = 'noop'
    # camera_name = 'camera3'
    # filter_obj = Source(source_name)
    # multiprocessing.Process(target=image_noop_node.run_node,
    #                         args=(filter_obj, camera_name)).start()
    # producer_wrappers.append(filter_obj.get_producer_wrapper())

    ## Launch nodes given source name:
    for source in args.source_name:
        source_name = source
        filter_obj = Source(source_name)
        multiprocessing.Process(
            target=docker_subsriber.run_node, args=(filter_obj, source_name)
        ).start()
        producer_wrappers.append(filter_obj.get_producer_wrapper())

    while True:
        client = WebsocketClient(server_addr, 9098, producer_wrappers, consumer)
        client.launch()
        logger.warning("Now trying to reconnect...")
        client.stop()
        time.sleep(5)


if __name__ == "__main__":
    main()
