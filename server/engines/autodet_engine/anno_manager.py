# SPDX-FileCopyrightText: 2021 Carnegie Mellon University
#
# SPDX-License-Identifier: Apache-2.0

import argparse
import glob
import json
import os
import time
from collections import defaultdict


class Anno_manager:
    def __init__(self, manual_anno_dir, neg_ratio=2, pos_need=10):
        self.pos_data_dict = {
            "images": [],
            "img_to_anns": defaultdict(list),
        }
        self.neg_data_dict = {
            "images": [],
            "img_to_anns": defaultdict(list),
        }

        self.img_index = {}
        self.pos_img_id_set = set()
        self.neg_img_id_set = set()
        self.img_id = 0
        self.ann_id = 0
        self.neg_ratio = neg_ratio
        self.pos_need = pos_need

        self.manual_anno_dir = manual_anno_dir
        file_list = sorted(
            glob.glob(os.path.join(self.manual_anno_dir, "annotations_*.json"))
        )
        if len(file_list) > 0:
            self.model_version = int(file_list[-1].split("/")[-1].split("_")[-1][:-5])
        else:
            self.model_version = 0
        print("model_version = ", self.model_version)

    def add_json(self, new_json_file):
        with open(new_json_file, "r") as new_anno_file:
            new_coco_dict = json.load(new_anno_file)

        for img_dict in new_coco_dict["images"]:
            self.img_id += 1
            new_img_id = self.img_id
            old_img_id = img_dict["id"]
            img_dict["id"] = new_img_id
            self.img_index[new_img_id] = img_dict

            for ann_dict in new_coco_dict["annotations"]:
                if ann_dict["image_id"] == old_img_id:
                    ann_dict["image_id"] = new_img_id

        for ann_dict in new_coco_dict["annotations"]:
            self.ann_id += 1
            new_ann_id = self.ann_id
            old_ann_id = ann_dict["id"]
            ann_dict["id"] = new_ann_id

            if ann_dict["category_id"] == 1:
                img_id = ann_dict["image_id"]
                self.pos_img_id_set.add(img_id)
                self.pos_data_dict["img_to_anns"][img_id].append(ann_dict)
            elif ann_dict["category_id"] == 2:
                img_id = ann_dict["image_id"]
                self.neg_img_id_set.add(img_id)
                self.neg_data_dict["img_to_anns"][img_id].append(ann_dict)
            else:
                raise NotImplementedError
        for img_id in self.pos_img_id_set:
            self.pos_data_dict["images"].append(self.img_index[img_id])
        self.pos_img_id_set = set()
        for img_id in self.neg_img_id_set:
            self.neg_data_dict["images"].append(self.img_index[img_id])
        self.neg_img_id_set = set()

        self.check_pos_num()
        os.remove(new_json_file)
        print(
            "\n{} has been added to cache.\nNow we have {} pos and {} neg".format(
                os.path.basename(new_json_file),
                len(self.pos_data_dict["images"]),
                len(self.neg_data_dict["images"]),
            )
        )

    def check_pos_num(self):
        while len(self.pos_data_dict["images"]) >= self.pos_need:
            self.dump_json()

    def dump_json(self):
        coco_data_dict = {
            "info": [],
            "licenses": [],
            "categories": [
                {"id": 1, "name": "positive", "supercategory": ""},
                {"id": 2, "name": "hard negative", "supercategory": ""},
            ],
            "images": [],
            "annotations": [],
        }
        #         print(self.pos_data_dict)
        for i in range(self.pos_need):
            img_dict = self.pos_data_dict["images"].pop(0)
            coco_data_dict["images"].append(img_dict)
            img_id = img_dict["id"]

            #             print("i = {}, id = {}".format(i,img_id))
            anno_list = self.pos_data_dict["img_to_anns"].pop(img_id)
            coco_data_dict["annotations"].extend(anno_list)

        neg_num = min(self.pos_need * self.neg_ratio, len(self.neg_data_dict["images"]))
        for i in range(neg_num):
            img_dict = self.neg_data_dict["images"].pop(0)
            coco_data_dict["images"].append(img_dict)
            img_id = img_dict["id"]

            anno_list = self.neg_data_dict["img_to_anns"].pop(img_id)
            coco_data_dict["annotations"].extend(anno_list)

        self.model_version += 1

        json_file_name = os.path.join(
            self.manual_anno_dir, "annotations_{}.json".format(self.model_version)
        )
        with open(json_file_name, "w") as f:
            json.dump(coco_data_dict, f)
            print(
                "{} has been saved, with 10 pos and {} neg".format(
                    os.path.basename(json_file_name), neg_num
                )
            )


def main(args):
    target_name = args.target_name
    task_dir = os.path.join("./autoDet_tasks/", target_name)
    os.makedirs(task_dir, exist_ok=True)
    manual_anno_dir = os.path.join(task_dir, "manual_anno")
    os.makedirs(manual_anno_dir, exist_ok=True)
    anno_manager = Anno_manager(manual_anno_dir, args.neg_ratio, args.pos_need)
    while True:
        file_list = glob.glob(os.path.join(manual_anno_dir, "cvat*.json"))
        file_list.sort(key=os.path.getmtime)
        for anno_file in file_list:
            anno_manager.add_json(anno_file)
        time.sleep(0.1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-t",
        "--target-name",
        default="anno_test",
        help="Set target name for the Auto-Detectron pipeline",
    )
    parser.add_argument(
        "-r",
        "--neg-ratio",
        type=int,
        default=3,
        help="--neg-ratio",
    )
    parser.add_argument(
        "-p",
        "--pos-need",
        type=int,
        default=10,
        help="--pos-need",
    )
    args = parser.parse_args()

    main(args)
