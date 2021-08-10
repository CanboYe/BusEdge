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
from exif import set_gps_location
from sensor_msgs.msg import CompressedImage, NavSatFix
from tqdm import tqdm


class Preprocessor:
    def __init__(self, args):
        self.args = args

        if os.path.basename(args.input).split(".")[-1] == "bag":
            self.bag_files = [args.input]
        elif os.path.isdir(args.input):
            self.bag_files = sorted(glob.glob(os.path.join(args.input, "*.bag")))
        else:
            raise ValueError("Cannot find the rosbag files.")

        self.output_dir = args.output
        os.makedirs(self.output_dir, exist_ok=True)

        self.save_gps = args.save_gps
        # self.topics = ["/fix"] if args.save_gps else []
        self.topics = ["/fix"]
        self.topics.append("/camera{}/image_raw/compressed".format(args.cam_id))
        # We don't support extrating multiple cameras for now.
        # for cam_id in args.cam_id:
        #     self.topics.append("/camera{}/image_raw/compressed".format(cam_id))

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
            # else:
            #     print("Not passed deduplicate, diff:{}".format(diff))
        return pass_flag

    def del_blur(self, current_frame):
        if current_frame.ndim == 3:
            current_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        blur_score = cv2.Laplacian(current_frame, cv2.CV_64F).var()
        pass_flag = blur_score > self.args.del_blur_thres
        # if not pass_flag:
        #     print("Not passed del_blur, score:{}".format(blur_score))
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

    def extract_images(self):
        for i, bag_file in enumerate(self.bag_files):
            print(
                "Now extracting the {}th rosbag {} for topics {}".format(
                    i + 1, bag_file, self.topics
                )
            )

            bag = rosbag.Bag(bag_file, "r")
            # info_dict = yaml.load(bag._get_yaml_info())
            # print("\nbag_info:{}\n\n".format(info_dict))
            for topic, msg, t in tqdm(bag.read_messages(topics=self.topics)):
                if "image_raw" in topic:
                    cv_img = self.bridge.compressed_imgmsg_to_cv2(
                        msg, desired_encoding="passthrough"
                    )
                    timestamp_sec = float(t.secs + t.nsecs / 1e9)
                    save_flag = self.filtering(cv_img, timestamp_sec)
                    if save_flag:
                        time_stamps = "_{:0>10d}_{:0>9d}".format(t.secs, t.nsecs)
                        image_filename = topic[1:8] + time_stamps + ".jpg"
                        image_dir = os.path.join(self.output_dir, image_filename)
                        cv2.imwrite(image_dir, cv_img)
                        if self.save_gps:
                            set_gps_location(
                                image_dir,
                                self.CUR_GPS.latitude,
                                self.CUR_GPS.longitude,
                                self.CUR_GPS.altitude,
                            )

                elif "fix" in topic:
                    self.CUR_GPS = msg
            bag.close()


def main(args):
    preprocessor = Preprocessor(args)
    preprocessor.extract_images()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract images and GPS from a rosbag."
    )
    parser.add_argument(
        "-i", "--input", default="./test.bag", help="Input ROS bag or folder"
    )
    parser.add_argument(
        "-c",
        "--cam-id",
        type=int,
        default=5,
        help="Select camera ID to extract",
    )
    parser.add_argument("-o", "--output", default="./output", help="Output dir")
    parser.add_argument(
        "-g",
        "--save-gps",
        action="store_true",
        help="Whether to save GPS as exif info of the images",
    )
    parser.add_argument(
        "--deduplicate",
        action="store_true",
        help="Whether to deduplicate the images",
    )
    parser.add_argument(
        "--del-blur",
        action="store_true",
        help="Whether to remove blurred images",
    )
    parser.add_argument(
        "--deduplicate-thres",
        type=int,
        default=15,
        help="Deduplcate threshold",
    )
    parser.add_argument(
        "--del-blur-thres",
        type=int,
        default=100,
        help="Deblur threshold",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=-1,
        help="Throttle at given frame rate",
    )
    parser.add_argument(
        "--gps-filtering",
        nargs=4,
        metavar=("Lat1", "Lon1", "Lat2", "Lon2"),
        type=float,
        default=None,
        help="Two GPS locations are needed for GPS filtering. \
            [Lat1, Lon1, Lat2, Lon2] in decimal degrees.\
            You could use GoogleMap to get the \
            upper-left and bottom-right corners)",
    )
    args = parser.parse_args()
    main(args)
