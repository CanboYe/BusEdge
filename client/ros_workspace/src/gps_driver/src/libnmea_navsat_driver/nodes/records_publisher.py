#!/usr/bin/python2

import rospy
from nmea_navsat_driver.msg import Acc, Vel
from sensor_msgs.msg import NavSatFix, NavSatStatus
from std_msgs.msg import Float64


class Records_Publisher:
    def __init__(self):
        # Create a ROS publisher
        self.gps_msg_publisher = rospy.Publisher("/fix", NavSatFix, queue_size=1)
        self.vel_msg_publisher = rospy.Publisher("/velocity", Vel, queue_size=1)
        self.acc_msg_publisher = rospy.Publisher("/acceleration", Acc, queue_size=1)

    def read_gps_file(self, file_dir="/reg/v/gps/gps"):
        with open(file_dir, "r") as file:
            line = file.readline()
            line = line.strip()

            msg = NavSatFix()
            msg.header.stamp = rospy.get_rostime()
            msg.header.frame_id = "gps"

            vel_msg = Vel()
            vel_msg.header.stamp = msg.header.stamp
            vel_msg.header.frame_id = "velocity"

            if line == "0.0, 0.0, 0.0, 0.0, 0.0, 0, 0" or len(line) == 0:
                msg.status.status = NavSatStatus.STATUS_NO_FIX  # -1
                rospy.logwarn("GPS unavailable.")
            # if line != '0.0, 0.0, 0.0, 0.0, 0.0, 0, 0' and len(line) != 0:
            else:
                fileds = [float(val) for val in line.split(",")]
                #             0         1          2        3      4         5            6
                # Format: Longitude, Latitude, Altitude, Speed, Heading, Event_value, Satellites
                msg.status.service = NavSatStatus.SERVICE_GPS
                if len(fileds) != 7:
                    rospy.logwarn("GPS file has wrong format.")
                if fileds[6] >= 1:
                    msg.status.status = NavSatStatus.STATUS_SBAS_FIX  # 1
                else:
                    msg.status.status = NavSatStatus.STATUS_FIX  # 0

                msg.latitude = self.latlon_convert(fileds[1])
                msg.longitude = self.latlon_convert(fileds[0])
                msg.altitude = fileds[2]
                msg.position_covariance_type = NavSatFix.COVARIANCE_TYPE_UNKNOWN

                vel_msg.velocity = fileds[3]
                vel_msg.heading = fileds[4]

        return msg, vel_msg

    def latlon_convert(self, data):
        # data should be ddmm.mmmm
        flag = 1 if data >= 0 else -1
        data = flag * data
        degrees = data // 100
        minutes = data - degrees * 100
        return (degrees + minutes / 60) * flag

    def publish_gps(self, event=None):
        msg, vel_msg = self.read_gps_file()
        self.gps_msg_publisher.publish(msg)
        self.vel_msg_publisher.publish(vel_msg)

    def read_acc_file(self, file_dir="/reg/v/acceleration/acceleration"):
        with open(file_dir, "r") as file:
            line = file.readline()
            line = line.strip()

            msg = Acc()
            msg.header.stamp = rospy.get_rostime()
            msg.header.frame_id = "acceleration"

            if len(line) != 0:
                fileds = [float(val) for val in line.split(",")]
                msg.x = fileds[0]
                msg.y = fileds[1]
                msg.z = fileds[2]
                msg.RMS = fileds[3]
            else:
                rospy.logwarn("Acceleration unavailable.")

        return msg

    def publish_acc(self, event=None):
        msg = self.read_acc_file()
        self.acc_msg_publisher.publish(msg)


def main():
    rospy.init_node("records_publisher")
    pub = Records_Publisher()
    rospy.Timer(rospy.Duration(1.0), pub.publish_gps)
    rospy.Timer(rospy.Duration(1.0), pub.publish_acc)
    # Don't forget this or else the program will exit
    rospy.spin()


# pub = Records_Publisher()
# print(pub.latlon_convert(4027.2336), pub.latlon_convert(-7956.5796))
