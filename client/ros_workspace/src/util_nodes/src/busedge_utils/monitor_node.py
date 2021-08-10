#!/usr/bin/python2

# SPDX-FileCopyrightText: 2021 Carnegie Mellon University
#
# SPDX-License-Identifier: Apache-2.0

# TODO: Update needed

from pathlib import Path

import rospy
from busedge_utils import email_sender, get_disk_status
from busedge_utils.msg import ErrMsg
from sensor_msgs.msg import NavSatFix, NavSatStatus
from std_msgs.msg import Float64


def readline_from(filename):

    with open(filename, "r") as f:
        return f.readline()


def write_file(lines, dir):
    with open(dir, "w") as f:
        f.write(lines)


class System_Monitor:
    def __init__(self):
        # Create a ROS publisher
        # self.gps_msg_publisher = rospy.Publisher("/fix", NavSatFix, queue_size=1)
        # self.vel_msg_publisher = rospy.Publisher("/velocity", Float64, queue_size=1)
        pass

    def heartbeat_cb(self, event=None):
        root_dir = "/home/albert/STATUS/"

        lines = "*" * 50 + "\n"
        strs = "Gabriel Status"
        lines += f"{strs:^50}" + "\n"
        lines += "*" * 50 + "\n"

        filename = root_dir + "gabriel_client_status"
        status = readline_from(filename)
        strs = "WebSocket Status: " + status
        lines += f"{strs:<25}"

        filename = root_dir + "gabriel_client_error"
        error = readline_from(filename)
        strs = "Error Counter: " + error
        lines += f"{strs:>25}\n"

        filter_type = "sign_filter"
        for cam_id in range(1, 6):
            source_name = filter_type + str(cam_id)
            # camera_name = 'camera' + str(cam_id)
            filename = root_dir + source_name + "_status"
            status = readline_from(filename)
            strs = source_name + ": " + status + "\n"
            lines += strs
        source_name = "gps"
        filename = root_dir + source_name + "_status"
        status = readline_from(filename)
        strs = "Trajectory" + ": " + status + "\n"
        lines += strs

        lines += "*" * 50 + "\n"
        strs = "Recorder Status"
        lines += f"{strs:^50}" + "\n"
        lines += "*" * 50 + "\n"

        # disk_stat = get_disk_status.disk_stat('.')
        disk_stat = get_disk_status.disk_stat(".")
        record_directory = Path("/home/albert/RECORDS")
        rosbag_size = (
            sum(f.stat().st_size for f in record_directory.glob("**/*") if f.is_file())
            / 1024.0
            / 1024
            / 1024
        )
        strs = "Storage: avail - {:.2f} GB;    rosbag - {:.2f} GB\n".format(
            disk_stat["free"], rosbag_size
        )
        lines += strs

        filename = root_dir + "data_usage"
        strs = "Network: " + readline_from(filename)
        lines += strs

        filename = root_dir + "all_status"
        write_file(lines, filename)
        return

    # def error_cb(self, data):
    #     print(data)
    #     msg = 'error email'
    #     email_sender.send(msg)
    #     return


def main():
    rospy.init_node("system_monitor")

    monitor = System_Monitor()
    rospy.Timer(rospy.Duration(1), monitor.heartbeat_cb)
    # rospy.Subscriber('/error_msg', ErrMsg, monitor.error_cb, queue_size=1)

    # Don't forget this or else the program will exit
    rospy.spin()


if __name__ == "__main__":
    main()
