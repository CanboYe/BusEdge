# SPDX-FileCopyrightText: 2021 Carnegie Mellon University
#
# SPDX-License-Identifier: Apache-2.0

import argparse
import logging
import os

from autodet_engine_livemap import AutoDetEngine
from gabriel_server.network_engine import engine_runner

IN_DOCKER = os.environ.get("AM_I_IN_A_DOCKER_CONTAINER", False)
if IN_DOCKER:
    DEFAULT_SOCKET_ADDR = "tcp://172.17.0.1:5555"
else:
    DEFAULT_SOCKET_ADDR = "tcp://localhost:5555"
DEFAULT_PORT = 9098
DEFAULT_SOURCE_NAME = "autoDet_target"
DEFAULT_TARGET_NAME = "target"
ONE_MINUTE = 60000
REQUEST_RETRIES = 1000

logging.basicConfig(level=logging.INFO)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-a", "--socket-addr", default=DEFAULT_SOCKET_ADDR, help="Set socket address"
    )
    parser.add_argument("--use-svm", action="store_true", help="Use SVM or not")
    parser.add_argument(
        "-n", "--num-cls", type=int, default=2, help="Number of classes"
    )
    parser.add_argument(
        "--source-name",
        default=DEFAULT_SOURCE_NAME,
        help="Set source name for this pipelines",
    )
    parser.add_argument(
        "-t",
        "--target-name",
        default=DEFAULT_TARGET_NAME,
        help="Set target name for the Auto-Detectron pipeline",
    )
    parser.add_argument(
        "-v",
        "--visualize",
        action="store_true",
        help="Whether to visualize the detection results",
    )
    parser.add_argument(
        "-l",
        "--use-livemap",
        action="store_true",
        help="Whether to insert the results to the LiveMap database",
    )
    parser.add_argument(
        "-w", "--win-pos", type=int, default=1, help="cv window position"
    )
    args = parser.parse_args()

    source_name = args.source_name
    autoDet_engine = AutoDetEngine(
        source_name,
        use_svm=args.use_svm,
        num_cls=args.num_cls,
        target_name=args.target_name,
        visualize=args.visualize,
        use_livemap=args.use_livemap,
        win_pos=args.win_pos,
    )

    engine_runner.run(
        engine=autoDet_engine,
        source_name=source_name,
        server_address=args.socket_addr,
        all_responses_required=False,
        timeout=ONE_MINUTE,
        request_retries=REQUEST_RETRIES,
    )


if __name__ == "__main__":
    main()
