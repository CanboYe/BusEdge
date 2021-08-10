# SPDX-FileCopyrightText: 2021 Carnegie Mellon University
#
# SPDX-License-Identifier: Apache-2.0

import argparse
import glob
import os
import signal
import threading
import time

import yaml
from core.client import CvatClient
from logzero import logger


def load_yaml(yaml_dir):
    with open(yaml_dir, "r") as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            logger.critical(exc)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-t",
        "--target-name",
        default="test",
        help="Set target name for the Delphi pipeline",
    )
    args = parser.parse_args()
    config = load_yaml("./cfg/config.yml")
    config["target_name"] = args.target_name
    print(config)
    client = CvatClient(config)

    try:
        client.start()
    except (KeyboardInterrupt, Exception):
        client.stop()
        pid = os.getpid()
        os.kill(pid, signal.SIGKILL)
