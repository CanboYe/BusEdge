# SPDX-FileCopyrightText: 2021 Carnegie Mellon University
#
# SPDX-License-Identifier: Apache-2.0

# -*- coding: utf-8 -*-

"""
Extract images and GPS by playing a rosbag and subscribing to the topics.
"""

import argparse
import os

import cv2
import rospy

# Don't need to build cv_bridge using python3 if we only use compressed_imgmsg_to_cv2()
from cv_bridge import CvBridge
from exif import set_gps_location
from sensor_msgs.msg import CompressedImage, NavSatFix

# Your own model
# from your_filter import YourFilter

CUR_GPS = NavSatFix()


def main(args):
    camera_name = "camera" + str(args.cam_id)
    output_dir = args.output
    save_gps = args.save_gps
    os.makedirs(output_dir, exist_ok=True)

    rospy.init_node(camera_name + "_filter_node")
    rospy.loginfo("Initialized filter node for " + camera_name)

    # Your own model
    # model_dir = "path_to_saved_model"
    # model = YourFilter(model_dir)

    image_sub = rospy.Subscriber(
        camera_name + "/image_raw/compressed",
        CompressedImage,
        img_callback,
        # callback_args=(model),
        callback_args=(output_dir, save_gps),
        queue_size=1,
        buff_size=2 ** 24,
    )
    gps_sub = rospy.Subscriber("/fix", NavSatFix, gps_callback, queue_size=1)
    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()


def img_callback(msg, args):
    global CUR_GPS

    # model = args[0]
    output_dir = args[0]
    save_gps = args[1]
    bridge = CvBridge()
    frame = bridge.compressed_imgmsg_to_cv2(
        msg, desired_encoding="passthrough"
    )  # BGR images

    # Your codes here to process Image data
    # frame = frame[:, :, ::-1]  # BGR to RGB
    # output = model(frame)

    # This is only for image extraction
    t = msg.header.stamp
    time_stamps = "{:0>10d}_{:0>9d}".format(t.secs, t.nsecs)
    image_filename = time_stamps + ".jpg"
    image_dir = os.path.join(output_dir, image_filename)
    cv2.imwrite(image_dir, frame)
    if save_gps:
        set_gps_location(
            image_dir, CUR_GPS.latitude, CUR_GPS.longitude, CUR_GPS.altitude
        )


def gps_callback(data):
    # Your codes here to process GPS data
    global CUR_GPS
    CUR_GPS = data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract images and GPS from a rosbag in an online manner."
    )
    parser.add_argument(
        "-c",
        "--cam-id",
        type=int,
        default=3,
        help="Select camera ID to extract",
    )
    parser.add_argument("-o", "--output", default="./output", help="Output dir")
    parser.add_argument(
        "-g",
        "--save-gps",
        action="store_true",
        help="Whether to save GPS as exif info of the images",
    )
    args = parser.parse_args()
    main(args)
