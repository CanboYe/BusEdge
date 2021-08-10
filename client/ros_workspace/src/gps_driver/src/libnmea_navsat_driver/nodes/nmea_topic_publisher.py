#!/usr/bin/python2

import rospy
from nmea_msgs.msg import Sentence


class NMEA_Publisher:
    def __init__(self):
        # Create a ROS publisher
        self.nmea_msg_publisher = rospy.Publisher(
            "/nmea_sentence", Sentence, queue_size=1
        )

    def read_gps_file(self, file_dir="/reg/v/gps/gprmc"):
        with open(file_dir, "r") as file:
            line = file.readline()

        return line

    def publish_nmea(self, event=None):
        data = self.read_gps_file()
        # print(data)
        sentence = Sentence()
        sentence.header.stamp = rospy.get_rostime()
        sentence.header.frame_id = "gps"
        sentence.sentence = data
        self.nmea_msg_publisher.publish(sentence)


def main():
    rospy.init_node("nmea_msg_publisher")

    nmea_pub = NMEA_Publisher()
    rospy.Timer(rospy.Duration(1.0), nmea_pub.publish_nmea)
    # Don't forget this or else the program will exit
    rospy.spin()
