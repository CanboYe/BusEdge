# SPDX-FileCopyrightText: 2021 Carnegie Mellon University
#
# SPDX-License-Identifier: Apache-2.0

import argparse
import logging
import os

from gabriel_server.network_engine import engine_runner
from your_engine import ExampleEngine

IN_DOCKER = os.environ.get("AM_I_IN_A_DOCKER_CONTAINER", False)
if IN_DOCKER:
    DEFAULT_SOCKET_ADDR = "tcp://172.17.0.1:5555"
else:
    DEFAULT_SOCKET_ADDR = "tcp://localhost:5555"
DEFAULT_PORT = 9098
DEFAULT_SOURCE_NAME = "sign_filter3"
ONE_MINUTE = 60000
REQUEST_RETRIES = 1000

logging.basicConfig(level=logging.INFO)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-a", "--socket-addr", default=DEFAULT_SOCKET_ADDR, help="Set socket address"
    )
    parser.add_argument(
        "-n",
        "--source-name",
        default=DEFAULT_SOURCE_NAME,
        help="Set source name for this pipeline",
    )
    args = parser.parse_args()

    example_engine = ExampleEngine(args.source_name)

    engine_runner.run(
        engine=example_engine,
        source_name=args.source_name,
        server_address=args.socket_addr,
        all_responses_required=False,
        timeout=ONE_MINUTE,
        request_retries=REQUEST_RETRIES,
    )


if __name__ == "__main__":
    main()
