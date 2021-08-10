# SPDX-FileCopyrightText: 2021 Carnegie Mellon University
#
# SPDX-License-Identifier: Apache-2.0

import argparse
import os
import time

import cv2
import rospy
from busedge_protocol import busedge_pb2
from cv_bridge import CvBridge
from delphi_filter import DelphiFilter
from gabriel_protocol import gabriel_pb2
from sensor_msgs.msg import CompressedImage, NavSatFix
from std_msgs.msg import UInt8MultiArray

CUR_GPS = NavSatFix()
# MODEL = DelphiFilter(feature_extractor='mobilenet_v2', svm_model_dir='./model/delphi_crop_ped_sign/svc_model.pkl')
# TODO: change to class
MODEL = DelphiFilter(
    feature_extractor="FRCNN_FBNet", svm_model_dir="", svm_threshold=0.8
)
MODEL_VERSION = 0
LAST_UPDATE = time.time()


def run_node(camera_name, source_name):
    global MODEL, MODEL_VERSION
    node_name = "{}_{}_node".format(source_name, camera_name)
    rospy.init_node(node_name, disable_signals=True)
    rospy.loginfo("Initialized node " + node_name)

    cam_id = camera_name[-1]
    pub = rospy.Publisher(source_name + cam_id, UInt8MultiArray, queue_size=1)

    svm_model_dir = "./models/" + source_name + "/svc_model.pkl"
    model_update_dir = "./models/" + source_name + "/update_flag"
    while not os.path.exists(svm_model_dir):
        if not rospy.is_shutdown():
            request_model(pub)
            time.sleep(5)
    MODEL.model.update_svm(svm_model_dir)
    MODEL_VERSION += 1
    check_model_update(model_update_dir)
    rospy.loginfo(
        "The svm classifier has got updated to version {}.".format(MODEL_VERSION)
    )

    image_sub = rospy.Subscriber(
        camera_name + "/image_raw/compressed",
        CompressedImage,
        img_callback,
        callback_args=(camera_name, pub, model_update_dir, svm_model_dir),
        queue_size=1,
        buff_size=2 ** 24,
    )
    gps_sub = rospy.Subscriber("/fix", NavSatFix, gps_callback, queue_size=1)

    # spin() simply keeps python from exiting until this node is stopped
    # rospy does not support spinonce()
    rospy.spin()


def check_model_update(model_update_dir):
    with open(model_update_dir, "r+") as f:
        flag = f.read().strip()
        f.seek(0)
        f.write("0")
    return True if flag == "1" else False


def request_model(pub):
    input_frame = gabriel_pb2.InputFrame()
    input_frame.payload_type = gabriel_pb2.PayloadType.OTHER
    serialized_message = input_frame.SerializeToString()

    pub_data = UInt8MultiArray()
    pub_data.data = serialized_message
    pub.publish(pub_data)

    rospy.loginfo("Sent model request to the server.")


def img_callback(image, args):
    global CUR_GPS, MODEL, MODEL_VERSION, LAST_UPDATE

    camera_name = args[0]
    pub = args[1]
    model_update_dir = args[2]
    svm_model_dir = args[3]

    if check_model_update(model_update_dir):
        MODEL.model.update_svm(svm_model_dir)
        MODEL_VERSION += 1
        rospy.loginfo(
            "The svm classifier has got updated to version {}.".format(MODEL_VERSION)
        )

    camera_id = int(camera_name[-1])

    engine_fields = busedge_pb2.EngineFields()
    engine_fields.gps_data.latitude = CUR_GPS.latitude
    engine_fields.gps_data.longitude = CUR_GPS.longitude
    engine_fields.gps_data.altitude = CUR_GPS.altitude

    bridge = CvBridge()
    frame = bridge.compressed_imgmsg_to_cv2(
        image, desired_encoding="passthrough"
    )  # BGR images
    frame = frame[:, :, ::-1]  # BGR to RGB
    frame_copy = frame.copy()

    method = "detection"
    if method == "detection":
        send_flag, results_bytes_to_send = MODEL.detect_and_send(frame_copy)
        if send_flag:
            _, jpeg_frame = cv2.imencode(".jpg", frame)
            input_frame = gabriel_pb2.InputFrame()
            input_frame.payload_type = gabriel_pb2.PayloadType.IMAGE
            input_frame.payloads.append(jpeg_frame.tobytes())

            secs = image.header.stamp.secs
            nsecs = image.header.stamp.nsecs
            time_stamps = "_{:0>10d}_{:0>9d}".format(secs, nsecs)
            image_filename = camera_name + time_stamps + ".jpg"
            engine_fields.image_filename = image_filename
            engine_fields.filter_results = results_bytes_to_send

            input_frame.extras.Pack(engine_fields)
            serialized_message = input_frame.SerializeToString()

            rospy.loginfo(
                "Sent detection results and raw image msg with size {:.2f} KB\n".format(
                    len(serialized_message) / 1024
                )
            )

            pub_data = UInt8MultiArray()
            pub_data.data = serialized_message
            pub.publish(pub_data)

            time.sleep(0.1)

    else:
        raise ValueError


def gps_callback(data):
    global CUR_GPS
    if data.status.status == -1:
        rospy.logdebug("Sign filter node cannot get valid GPS data")
    else:
        CUR_GPS = data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--camera-id",
        type=int,
        default=5,
        help="Set the camera id for the Delphi pipeline",
    )
    parser.add_argument(
        "-n",
        "--source-name",
        default="trash_can",
        help="Set source name for the Delphi pipeline",
    )
    args = parser.parse_args()
    try:
        run_node("camera" + str(args.camera_id), args.source_name)
    except (KeyboardInterrupt, Exception):
        print("Shutting down...")
        rospy.signal_shutdown("Manual shutdown")
