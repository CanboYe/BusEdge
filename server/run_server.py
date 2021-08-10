# SPDX-FileCopyrightText: 2021 Carnegie Mellon University
#
# SPDX-License-Identifier: Apache-2.0

import argparse
import logging

from gabriel_server.network_engine import server_runner

DEFAULT_PORT = 9098
DEFAULT_NUM_TOKENS = 2
DEFAULT_SOCKET_ADDR = "tcp://*:5555"
INPUT_QUEUE_MAX_SIZE = 60

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
DATE_FORMAT = "%m/%d-%H:%M:%S"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, datefmt=DATE_FORMAT)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-t", "--tokens", type=int, default=DEFAULT_NUM_TOKENS, help="number of tokens"
    )
    parser.add_argument(
        "-p", "--port", type=int, default=DEFAULT_PORT, help="Set port number"
    )
    parser.add_argument(
        "-a", "--socket-addr", default=DEFAULT_SOCKET_ADDR, help="Set socket address"
    )
    args = parser.parse_args()

    server_runner.run(
        args.port, args.socket_addr, args.tokens, INPUT_QUEUE_MAX_SIZE, timeout=20
    )


if __name__ == "__main__":
    main()
