#!/usr/bin/python2

# SPDX-FileCopyrightText: 2021 Carnegie Mellon University
#
# SPDX-License-Identifier: Apache-2.0

import cv2
import rospy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image

bridge = CvBridge()


def image_callback(msg, args):
    # print('get an image')
    filename_prefix = str(args[0])
    try:
        # Convert your ROS Image message to OpenCV2
        cv2_img = bridge.imgmsg_to_cv2(msg, "bgr8")
    except CvBridgeError as e:
        print(e)
    else:
        # Save your OpenCV2 image as a jpeg
        secs = msg.header.stamp.secs
        nsecs = msg.header.stamp.nsecs
        time = "{:0>10d}_{:0>9d}".format(secs, nsecs)
        cv2.imwrite(filename_prefix + time + ".jpg", cv2_img)
        # rospy.sleep(1)


def main():
    rospy.init_node("image_saver")
    rospy.loginfo("Initialized node image_saver")
    image_topic = rospy.get_param("~image", "/camera/image_raw")
    filename_prefix = rospy.get_param("~filename", "image_")
    # Set up your subscriber and define its callback
    image_sub = rospy.Subscriber(
        image_topic,
        Image,
        image_callback,
        callback_args=(filename_prefix,),
        queue_size=None,
        buff_size=2 ** 24,
    )
    # Spin until ctrl + c
    rospy.spin()


if __name__ == "__main__":
    main()
