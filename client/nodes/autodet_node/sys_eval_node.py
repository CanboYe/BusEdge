# SPDX-FileCopyrightText: 2021 Carnegie Mellon University
#
# SPDX-License-Identifier: Apache-2.0

import argparse
import os
import time
from glob import glob

import cv2
import rospy
from cv_bridge import CvBridge
from delphi_filter import DelphiFilter
from gabriel_protocol import gabriel_pb2
from protocol import busedge_delphi_pb2
from sensor_msgs.msg import CompressedImage, NavSatFix
from std_msgs.msg import UInt8MultiArray

CUR_GPS = NavSatFix()
MODEL = DelphiFilter(
    feature_extractor="FRCNN_FBNet", svm_model_dir="", svm_threshold=0.6
)
MODEL_VERSION = 0
LAST_UPDATE = time.time()


def run_node(camera_name, source_name, img_folder, use_model):
    global MODEL, MODEL_VERSION
    node_name = "{}_{}_node".format(source_name, camera_name)
    rospy.init_node(node_name, disable_signals=True)
    rospy.loginfo("Initialized node " + node_name)

    pub = rospy.Publisher(source_name, UInt8MultiArray, queue_size=1)

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

    read_imgs(pub, model_update_dir, svm_model_dir, img_folder, use_model)


def read_imgs(pub, model_update_dir, svm_model_dir, img_folder, use_model):
    global MODEL, MODEL_VERSION, LAST_UPDATE
    sent_img_count = 0
    img_list = sorted(glob(os.path.join(img_folder, "*.jpg")))
    img_len = len(img_list)

    for img_i, img_dir in enumerate(img_list):
        image_filename = os.path.basename(img_dir)
        frame = cv2.imread(img_dir)
        frame = frame[:, :, ::-1]  # BGR to RGB
        frame_copy = frame.copy()

        if check_model_update(model_update_dir):
            MODEL.model.update_svm(svm_model_dir)
            MODEL_VERSION += 1
            rospy.loginfo(
                "The svm classifier has got updated to version {}.".format(
                    MODEL_VERSION
                )
            )
        delta = time.time() - LAST_UPDATE
        if delta > 15:
            request_model(pub)
            LAST_UPDATE = time.time()

        save_img_folder = "./bus_sent"
        os.makedirs(save_img_folder, exist_ok=True)
        save_img_dir = os.path.join(save_img_folder, image_filename)

        if use_model:
            send_flag, results_bytes_to_send = MODEL.detect_and_send(
                frame_copy, save_img_dir
            )
        else:
            send_flag, results_bytes_to_send = True, None

        if send_flag:
            sent_img_count += 1
            print("Sent_img_count = ", sent_img_count)
            _, jpeg_frame = cv2.imencode(".jpg", frame)
            input_frame = gabriel_pb2.InputFrame()
            input_frame.payload_type = gabriel_pb2.PayloadType.IMAGE
            input_frame.payloads.append(jpeg_frame.tobytes())

            engine_fields = busedge_delphi_pb2.EngineFields()
            engine_fields.gps_data.latitude = 0
            engine_fields.gps_data.longitude = 0
            engine_fields.gps_data.altitude = 0
            engine_fields.image_filename = image_filename
            if results_bytes_to_send:
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

        # window_name = "Raw Data"
        # cv2.namedWindow(
        #     window_name,
        #     cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_NORMAL,
        # )
        # width, height = 480, 270
        # cv2.moveWindow(window_name, 2000, 80)
        # cv2.resizeWindow(window_name, width, height)
        # cv2.imshow("Raw Data", frame[:, :, ::-1])

        # if sent_img_count == 1:
        #     cv2.waitKey(0)
        #     sent_img_count += 1
        # if img_i == img_len-1:
        #     cv2.waitKey(0)
        # if send_flag:
        #     if cv2.waitKey(1) & 0xFF == ord('q'):
        #         break
        # else:
        #     if cv2.waitKey(1) & 0xFF == ord('q'):
        #         break
        # time.sleep(0.1)


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
        "-t",
        "--target-name",
        default="sys_eval",
        help="Set target name for the Delphi pipeline",
    )
    parser.add_argument(
        "-i",
        "--input-folder",
        default="./cloudy_downtown",
        help="Set input folder",
    )
    parser.add_argument(
        "-m", "--use-model", action="store_false", help="Use model or not"
    )
    args = parser.parse_args()

    try:
        run_node(
            "camera" + str(args.camera_id),
            "autoDet_" + args.target_name,
            args.input_folder,
            args.use_model,
        )
    except (KeyboardInterrupt, Exception):
        print("Shutting down...")
        rospy.signal_shutdown("Manual shutdown")
