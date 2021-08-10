# SPDX-FileCopyrightText: 2021 Carnegie Mellon University
#
# SPDX-License-Identifier: Apache-2.0

import time

import cv2
import rospy
from busedge_protocol import busedge_pb2
from cv_bridge import CvBridge, CvBridgeError
from gabriel_protocol import gabriel_pb2
from sensor_msgs.msg import CompressedImage, Image, NavSatFix

LAST_GPS = NavSatFix()
CUR_GPS = NavSatFix()


def run_node(noop_filter, camera_name):
    rospy.init_node(camera_name + "noop_node")
    rospy.loginfo("Initialized node noop for " + camera_name)
    image_sub = rospy.Subscriber(
        camera_name + "/image_raw/compressed",
        CompressedImage,
        img_callback,
        callback_args=(noop_filter, camera_name),
        queue_size=1,
        buff_size=2 ** 24,
    )
    gps_sub = rospy.Subscriber("/fix", NavSatFix, gps_callback, queue_size=1)
    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()


def img_callback(image, args):
    global CUR_GPS

    noop_filter = args[0]
    camera_name = args[1]
    bridge = CvBridge()
    # frame = bridge.imgmsg_to_cv2(image, "rgb8")
    frame = bridge.compressed_imgmsg_to_cv2(
        image, desired_encoding="passthrough"
    )  # BGR images
    frame = frame[:, :, ::-1]  # BGR to RGB

    _, jpeg_frame = cv2.imencode(".jpg", frame)
    input_frame = gabriel_pb2.InputFrame()
    input_frame.payload_type = gabriel_pb2.PayloadType.IMAGE
    input_frame.payloads.append(jpeg_frame.tobytes())

    # engine_fields = navlab_pb2.EngineFields()
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

    noop_filter.send(input_frame)
    # rospy.loginfo('I heard image at %.4f, %.4f, %.4f and packed it to from_client.', CUR_GPS.latitude, CUR_GPS.longitude, CUR_GPS.altitude)

    show_flag = True
    if show_flag:
        cv2.namedWindow("Raw images", 0)
        cv2.imshow("Raw images", frame[:, :, ::-1])
        cv2.waitKey(1)
    time.sleep(0.1)


def gps_callback(data):
    global LAST_GPS, CUR_GPS
    if data.status.status == -1:
        print("Unable to get a fix on the location.")
    else:
        LAST_GPS = CUR_GPS
        CUR_GPS = data
