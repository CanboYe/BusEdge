# SPDX-FileCopyrightText: 2021 Carnegie Mellon University
#
# SPDX-License-Identifier: Apache-2.0

import logging

import cv2
from busedge_protocol import busedge_pb2
from gabriel_protocol import gabriel_pb2
from sign_filter import SignFilter

logger = logging.getLogger(__name__)

import argparse
import multiprocessing
import time

import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage, Image, NavSatFix
from std_msgs.msg import UInt8MultiArray

DEFAULT_SOURCE_NAME = "sign_filter3"
CUR_GPS = NavSatFix()


def run_node(source_name):
    cam_id = source_name[-1]
    camera_name = "camera" + cam_id
    rospy.init_node(camera_name + "_sign_filter_node")
    rospy.loginfo("Initialized node sign_filter for " + camera_name)
    model_dir = "./model/ssd_mobilenet_v1_mtsd_hunter/saved_model"
    model = SignFilter(model_dir)

    pub = rospy.Publisher(source_name, UInt8MultiArray, queue_size=1)
    image_sub = rospy.Subscriber(
        camera_name + "/image_raw/compressed",
        CompressedImage,
        img_callback,
        callback_args=(model, camera_name, pub),
        queue_size=1,
        buff_size=2 ** 24,
    )
    gps_sub = rospy.Subscriber("/fix", NavSatFix, gps_callback, queue_size=1)

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()


def img_callback(image, args):
    global CUR_GPS

    model = args[0]
    camera_name = args[1]
    pub = args[2]

    camera_id = int(camera_name[-1])

    bridge = CvBridge()
    frame = bridge.compressed_imgmsg_to_cv2(
        image, desired_encoding="passthrough"
    )  # BGR images
    frame = frame[:, :, ::-1]  # BGR to RGB
    frame_copy = frame.copy()

    # FILTER
    # send_flag = model.send(frame_copy, show_flag = True)
    min_score_thresh = 0.75
    output_dict = model.detect(frame_copy, min_score_thresh)
    send_flag = output_dict["num_detections"] > 0

    if send_flag == True:
        _, jpeg_frame = cv2.imencode(".jpg", frame)

        input_frame = gabriel_pb2.InputFrame()
        input_frame.payload_type = gabriel_pb2.PayloadType.IMAGE
        input_frame.payloads.append(jpeg_frame.tostring())

        engine_fields = busedge_pb2.EngineFields()
        engine_fields.gps_data.latitude = CUR_GPS.latitude
        engine_fields.gps_data.longitude = CUR_GPS.longitude
        engine_fields.gps_data.altitude = CUR_GPS.altitude

        secs = image.header.stamp.secs
        nsecs = image.header.stamp.nsecs
        time_stamps = "_{:0>10d}_{:0>9d}".format(secs, nsecs)
        image_filename = camera_name + time_stamps + ".jpg"
        engine_fields.image_filename = image_filename

        input_frame.extras.Pack(engine_fields)
        serialized_message = input_frame.SerializeToString()

        rospy.loginfo(
            "Sent image msg with size {:.2f} KB".format(len(serialized_message) / 1024)
        )

        pub_data = UInt8MultiArray()
        pub_data.data = serialized_message
        pub.publish(pub_data)

        time.sleep(0.1)
    else:
        pass


def gps_callback(data):
    global CUR_GPS
    if data.status.status == -1:
        rospy.logdebug("Sign filter node cannot get valid GPS data")
    else:
        CUR_GPS = data


if __name__ == "__main__":
    # run_node('camera3')

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-n",
        "--source-name",
        nargs="+",
        default=[DEFAULT_SOURCE_NAME],
        help="Set source name for this pipeline",
    )
    args = parser.parse_args()

    for source in args.source_name:
        multiprocessing.Process(target=run_node, args=(source,)).start()
