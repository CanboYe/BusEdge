# SPDX-FileCopyrightText: 2021 Carnegie Mellon University
#
# SPDX-License-Identifier: Apache-2.0

import json
import os

import numpy as np


class PseudoAnnotator:
    def __init__(self, categories, output_json_dir):
        self.output_json_dir = output_json_dir
        os.makedirs(os.path.dirname(self.output_json_dir), exist_ok=True)
        self.image_id = 0
        self.anno_id = 0
        categories_dict = [
            {"id": i + 1, "name": cls_name, "supercategory": ""}
            for i, cls_name in enumerate(categories)
        ]
        self.coco_anno_dict = {
            "info": [],
            "licenses": [],
            "categories": categories_dict,
            "images": [],
            "annotations": [],
        }

    def parse_instances(self, instances):
        width, height = instances.image_size
        classes = instances.pred_classes
        scores = instances.scores
        bboxes = instances.pred_boxes
        # masks = instances['instances'].pred_masks

        classes = classes.numpy()
        scores = scores.numpy()
        bboxes = bboxes.tensor.numpy()
        # masks = masks.numpy()

        scores = np.around(scores, decimals=2).astype(float)
        bboxes = np.around(bboxes, decimals=2).astype(float)
        return bboxes, classes, scores, width, height

    def add_pseudo_annotations(self, img_name, instances):
        boxes, classes, scores, width, height = self.parse_instances(instances)

        self.image_id += 1
        image_dict = {
            "id": self.image_id,
            "width": width,
            "height": height,
            "file_name": img_name,
        }
        self.coco_anno_dict["images"].append(image_dict)
        anno_dicts = []
        for i in range(boxes.shape[0]):
            self.anno_id += 1
            box = boxes[i]  # (x1, y1, x2, y2)
            area = (box[2] - box[0]) * (box[3] - box[1])
            box_coco = [box[0], box[1], box[2] - box[0], box[3] - box[1]]
            anno_dict = {
                "id": self.anno_id,
                "image_id": self.image_id,
                "category_id": classes[i].item() + 1,
                "area": area,
                "bbox": box_coco,
                "iscrowd": 0,
                #                 "score": scores[i].item(),
            }
            anno_dicts.append(anno_dict)
        self.coco_anno_dict["annotations"].extend(anno_dicts)

    def dump_json(self):
        pseudo_anno_file = open(self.output_json_dir, "w")
        json.dump(self.coco_anno_dict, pseudo_anno_file)

    def launch(self, img_name, instances):
        self.add_pseudo_annotations(img_name, instances)
        self.dump_json()
