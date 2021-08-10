# SPDX-FileCopyrightText: 2021 Carnegie Mellon University
#
# SPDX-License-Identifier: Apache-2.0

import argparse
import glob
import os
import time

from detector import AutoDetector


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-t",
        "--target-name",
        default="trash_can",
        help="Set target name for the Auto-Detectron pipeline",
    )
    parser.add_argument("--use-svm", action="store_true", help="Use SVM or not")
    parser.add_argument(
        "-n", "--num-cls", type=int, default=2, help="Number of classes"
    )
    args = parser.parse_args()

    target_name = args.target_name
    detector = AutoDetector(target_name, num_cls=args.num_cls, use_svm=args.use_svm)

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
                if args.use_svm:
                    detector.retrain_finetune_svm(basename)
                else:
                    detector.retrain_finetune(basename)
        time.sleep(1)


if __name__ == "__main__":
    main()
