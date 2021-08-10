# SPDX-FileCopyrightText: 2021 Carnegie Mellon University
#
# SPDX-License-Identifier: Apache-2.0

import logging
import time

import rospy
from busedge_protocol import busedge_pb2
from gabriel_protocol import gabriel_pb2
from sensor_msgs.msg import NavSatFix

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def run_node(gps_noop_filter):
    rospy.init_node("gps_noop_node")
    rospy.loginfo("Initialized node gps_noop")
    # image_sub = rospy.Subscriber("camera/image_raw", Image, img_callback_noop, callback_args=(noop_filter,), queue_size=1, buff_size=2**24)
    gps_sub = rospy.Subscriber(
        "/fix",
        NavSatFix,
        trajectory_callback,
        callback_args=(gps_noop_filter,),
        queue_size=1,
    )
    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()


def trajectory_callback(data, args):
    if data.status.status == -1:
        rospy.logdebug("Trajectory node cannot get valid GPS data")
    else:
        gps_noop_filter = args[0]

        input_frame = gabriel_pb2.InputFrame()
        input_frame.payload_type = gabriel_pb2.PayloadType.OTHER

        engine_fields = busedge_pb2.EngineFields()
        engine_fields.gps_data.latitude = data.latitude
        engine_fields.gps_data.longitude = data.longitude
        engine_fields.gps_data.altitude = data.altitude
        input_frame.extras.Pack(engine_fields)

        gps_noop_filter.send(input_frame)
        # rospy.loginfo('Sent GPS %.4f, %.4f, %.4f.', data.latitude, data.longitude, data.altitude)
        time.sleep(0.1)
