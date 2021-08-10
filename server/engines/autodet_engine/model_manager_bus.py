# SPDX-FileCopyrightText: 2021 Carnegie Mellon University
#
# SPDX-License-Identifier: Apache-2.0

import argparse
import glob
import os
import time

from detector_fbnet import AutoDetector


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-t",
        "--target-name",
        default="trash_can",
        help="Set target name for the Auto-Detectron pipeline",
    )
    args = parser.parse_args()

    target_name = args.target_name
    detector = AutoDetector(target_name)

    task_dir = os.path.join("./autoDet_tasks/", target_name)
    os.makedirs(task_dir, exist_ok=True)
    manual_anno_dir = os.path.join(task_dir, "manual_anno")
    os.makedirs(manual_anno_dir, exist_ok=True)

    while True:
        model_version = detector.model_version
        file_list = glob.glob(os.path.join(manual_anno_dir, "*.json"))
        for anno_file in file_list:
            basename = os.path.basename(anno_file)
            if basename == "annotations_{}.json".format(model_version + 1):
                print("Now reading anno from {}".format(anno_file))
                detector.train_svc(basename)
        time.sleep(1)


if __name__ == "__main__":
    main()
