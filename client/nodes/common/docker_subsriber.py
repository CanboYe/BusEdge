# SPDX-FileCopyrightText: 2021 Carnegie Mellon University
#
# SPDX-License-Identifier: Apache-2.0

import cv2
import numpy as np
import rospy
from gabriel_protocol import gabriel_pb2
from std_msgs.msg import UInt8MultiArray


def run_node(client_filter, source_name):
    rospy.init_node(source_name + "_subscriber_node")
    rospy.loginfo("Initialized subscriber node for " + source_name)
    sub = rospy.Subscriber(
        source_name,
        UInt8MultiArray,
        callback,
        callback_args=(client_filter,),
        queue_size=1,
        buff_size=2 ** 24,
    )
    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()


def callback(data, args):
    client_filter = args[0]
    serialized_message = data.data

    # client_filter.send_serialized(serialized_message)
    # TODO: this is inefficient becuase we deserialize the binary data,
    #       need to either modify the gabriel library or change the way
    #       we save the extra fields.
    input_frame = gabriel_pb2.InputFrame()
    input_frame.ParseFromString(serialized_message)
    client_filter.send(input_frame)
