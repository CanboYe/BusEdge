# SPDX-FileCopyrightText: 2021 Carnegie Mellon University
#
# SPDX-License-Identifier: Apache-2.0

import time

import cv2
import rospy
from cv_bridge import CvBridge
from gabriel_protocol import gabriel_pb2
from protocol import navlab_pb2
from sensor_msgs.msg import CompressedImage, NavSatFix
from std_msgs.msg import UInt8MultiArray
from your_filter import YourFilter

CUR_GPS = NavSatFix()


def run_node(camera_name):
    rospy.init_node(camera_name + "_filter_node")
    rospy.loginfo("Initialized filter node for " + camera_name)

    # Your own model
    model_dir = "path_to_saved_model"
    model = YourFilter(model_dir)

    pub = rospy.Publisher("sign_filter", UInt8MultiArray, queue_size=1)
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

    min_score_thresh = 0.75
    output_dict = model.detect(frame_copy, min_score_thresh)
    send_flag = output_dict["num_detections"] > 0

    if send_flag == True:
        # save the results as the self-defined protobuf format and serilize it.
        _, jpeg_frame = cv2.imencode(".jpg", frame)

        input_frame = gabriel_pb2.InputFrame()
        input_frame.payload_type = gabriel_pb2.PayloadType.IMAGE
        input_frame.payloads.append(jpeg_frame.tostring())

        engine_fields = navlab_pb2.EngineFields()
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

        pub_data = UInt8MultiArray()
        pub_data.data = serialized_message
        pub.publish(pub_data)

        time.sleep(0.1)
    else:
        pass


def gps_callback(data):
    global CUR_GPS
    if data.status.status == -1:
        rospy.logdebug("Cannot get valid GPS data")
    else:
        CUR_GPS = data


if __name__ == "__main__":
    run_node("camera3")
