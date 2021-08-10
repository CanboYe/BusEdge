# SPDX-FileCopyrightText: 2021 Carnegie Mellon University
#
# SPDX-License-Identifier: Apache-2.0

import argparse
import time

import cv2
import numpy as np
import rospy
from busedge_utils.msg import FeatureMsg, PreprocessedMsg
from cv_bridge import CvBridge, CvBridgeError
from feature_extractor import FeatExtractor
from sensor_msgs.msg import CompressedImage, Image, NavSatFix
from std_msgs.msg import Float32MultiArray, MultiArrayDimension


def run_node(camera_name, model_type="FRCNN_FBNet"):
    rospy.init_node(camera_name + "_featExtractor_node")
    rospy.loginfo("Initialized featExtractor node for " + camera_name)
    cam_id = camera_name[-1]
    pub_img_feat = rospy.Publisher("image_feature" + cam_id, FeatureMsg, queue_size=1)
    pub_box_feat = rospy.Publisher("box_feature" + cam_id, FeatureMsg, queue_size=1)

    feat_extractor = FeatExtractor(feature_extractor=model_type)

    preprocessed_img_sub = rospy.Subscriber(
        "preprocessed" + cam_id,
        PreprocessedMsg,
        preprocessed_img_callback,
        callback_args=(camera_name, pub_img_feat, pub_box_feat, feat_extractor),
        queue_size=1,
        buff_size=2 ** 24,
    )
    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()


def preprocessed_img_callback(msg, args):
    camera_name = args[0]
    pub_img_feat = args[1]
    pub_box_feat = args[2]
    feat_extractor = args[3]

    bridge = CvBridge()
    image_msg = msg.image
    frame = bridge.compressed_imgmsg_to_cv2(
        image_msg, desired_encoding="passthrough"
    )  # BGR images
    frame = frame[:, :, ::-1]  # BGR to RGB

    show_raw = True
    if show_raw:
        cv2.namedWindow("Raw images for feature extractor", 0)
        cv2.imshow("Raw images for feature extractor", frame[:, :, ::-1])
        cv2.waitKey(1)

    img_feat = feat_extractor.extract_features(frame)

    # print("img_feat: ", img_feat.shape) # [channels, height, width] = [112, 30, 54]
    # print("img_feat[66, 10, 10]: ", img_feat[66, 10, 10])
    # print("box_feat: ", box_feat.shape) # [proposal_num, feat_vec_len] = [p, 200], p_max=500
    # print("GPS: {}, {}, {} with ts {}".format(msg.lat, msg.lon, msg.alt, msg.image_ts))

    ## publish image feature map
    img_feat_msg = FeatureMsg()
    img_feat_msg.image_ts = msg.image_ts
    img_feat_msg.lat = msg.lat
    img_feat_msg.lon = msg.lon
    img_feat_msg.alt = msg.alt

    matrix_msg = Float32MultiArray()

    # This is almost always zero there is no empty padding at the start of your data
    matrix_msg.layout.data_offset = 0
    matrix_msg.layout.dim = [
        MultiArrayDimension(),
        MultiArrayDimension(),
        MultiArrayDimension(),
    ]

    matrix_msg.layout.dim[0].label = "channels"
    matrix_msg.layout.dim[0].size = img_feat.shape[0]
    matrix_msg.layout.dim[0].stride = (
        img_feat.shape[0] * img_feat.shape[1] * img_feat.shape[2]
    )
    matrix_msg.layout.dim[1].label = "height"
    matrix_msg.layout.dim[1].size = img_feat.shape[1]
    matrix_msg.layout.dim[1].stride = img_feat.shape[1] * img_feat.shape[2]
    matrix_msg.layout.dim[2].label = "width"
    matrix_msg.layout.dim[2].size = img_feat.shape[2]
    matrix_msg.layout.dim[2].stride = img_feat.shape[2]

    matrix_msg.data = img_feat.flatten(order="C")  # flattern in row-major
    img_feat_msg.feature = matrix_msg
    pub_img_feat.publish(img_feat_msg)

    # ## publish bounding box feature map
    # box_feat_msg = FeatureMsg()
    # box_feat_msg.image_ts = msg.image_ts
    # box_feat_msg.lat = msg.lat
    # box_feat_msg.lon = msg.lon
    # box_feat_msg.alt = msg.alt

    # matrix_msg = Float32MultiArray()

    # # This is almost always zero there is no empty padding at the start of your data
    # matrix_msg.layout.data_offset = 0

    # # create two dimensions in the dim array
    # matrix_msg.layout.dim = [MultiArrayDimension(), MultiArrayDimension()]

    # # dim[0] is the vertical dimension of your matrix
    # matrix_msg.layout.dim[0].label = "proposal_num"
    # matrix_msg.layout.dim[0].size = box_feat.shape[0]
    # matrix_msg.layout.dim[0].stride = box_feat.shape[0] * box_feat.shape[1]
    # # dim[1] is the horizontal dimension of your matrix
    # matrix_msg.layout.dim[1].label = "feature_dim"
    # matrix_msg.layout.dim[1].size = box_feat.shape[1]
    # matrix_msg.layout.dim[1].stride = box_feat.shape[1]

    # matrix_msg.data = box_feat.flatten(order="C") # flattern in row-major

    # box_feat_msg.feature = matrix_msg
    # pub_box_feat.publish(box_feat_msg)


def main(args):
    run_node("camera" + str(args.cam_id), args.model_type)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Feature Extractor Node")

    parser.add_argument(
        "-c",
        "--cam-id",
        type=int,
        default=5,
        help="Select camera ID to extract",
    )
    parser.add_argument("-m", "--model-type", default="FRCNN_FBNet", help="Model Type")

    args = parser.parse_args()
    main(args)
