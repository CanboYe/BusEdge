# SPDX-FileCopyrightText: 2021 Carnegie Mellon University
#
# SPDX-License-Identifier: Apache-2.0

# -*- coding: utf-8 -*-

"""
Extract images and GPS by directly reading a rosbag.
"""

import argparse
import glob
import os

import cv2
import numpy as np
import rosbag
import yaml
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage, NavSatFix
from tqdm import tqdm


class Preprocessor:
    def __init__(self, args):
        self.args = args

        if args.output:
            self.output_dir = args.output
            os.makedirs(self.output_dir, exist_ok=True)

        self.topics = ["/fix"]
        self.topics.append("/camera{}/image_raw/compressed".format(args.cam_id))

        self.bridge = CvBridge()
        self.CUR_GPS = NavSatFix()
        self.last_frame = None
        self.last_ts = None
        if self.args.fps != -1:
            self.frame_interval = 1.0 / self.args.fps

    def deduplicate(self, current_frame):
        if self.last_frame is None:
            self.last_frame = current_frame
            pass_flag = True
        else:
            diff = np.mean(cv2.absdiff(current_frame, self.last_frame))
            pass_flag = diff > self.args.deduplicate_thres
            if pass_flag:
                self.last_frame = current_frame
            else:
                print("Not passed deduplicate, diff:{}".format(diff))
        return pass_flag

    def del_blur(self, current_frame):
        if current_frame.ndim == 3:
            current_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        blur_score = cv2.Laplacian(current_frame, cv2.CV_64F).var()
        pass_flag = blur_score > self.args.del_blur_thres
        if not pass_flag:
            print("Not passed del_blur, score:{}".format(blur_score))
        return pass_flag

    def throttle_fps(self, current_ts):
        if self.last_ts is None:
            self.last_ts = current_ts
            pass_flag = True
        else:
            interval = current_ts - self.last_ts
            pass_flag = interval > self.frame_interval
            if pass_flag:
                self.last_ts = current_ts
        return pass_flag

    def gps_filtering(self):

        lat_array = np.array(
            [
                self.CUR_GPS.latitude,
                self.args.gps_filtering[0],
                self.args.gps_filtering[2],
            ]
        )
        lon_array = np.array(
            [
                self.CUR_GPS.longitude,
                self.args.gps_filtering[1],
                self.args.gps_filtering[3],
            ]
        )
        lat_within = np.argsort(lat_array)[1] == 0
        lon_within = np.argsort(lon_array)[1] == 0
        # print("{}, {}".format(self.CUR_GPS.latitude, self.CUR_GPS.longitude))
        return lat_within and lon_within

    def filtering(self, current_frame, timestamp_sec):
        pass_flag = True
        if self.args.gps_filtering is not None:
            pass_flag = self.gps_filtering()
            if pass_flag == False:
                return False
        if self.args.fps != -1:
            pass_flag = self.throttle_fps(timestamp_sec)
            if pass_flag == False:
                return False
        if self.args.deduplicate:
            pass_flag = self.deduplicate(current_frame)
            if pass_flag == False:
                return False
        if self.args.del_blur:
            pass_flag = self.del_blur(current_frame)
            if pass_flag == False:
                return False
        return pass_flag
