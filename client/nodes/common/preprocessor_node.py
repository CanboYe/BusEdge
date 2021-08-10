# SPDX-FileCopyrightText: 2021 Carnegie Mellon University
#
# SPDX-License-Identifier: Apache-2.0

import argparse
import time

import cv2
import rospy
from busedge_utils.msg import PreprocessedMsg
from cv_bridge import CvBridge, CvBridgeError
from preprocessor import Preprocessor
from sensor_msgs.msg import CompressedImage, Image, NavSatFix

LAST_GPS = NavSatFix()
CUR_GPS = NavSatFix()
send_frame = 0
send_frame_raw = 0


def run_node(processor, camera_name):
    rospy.init_node(camera_name + "_preprocessor_node")
    rospy.loginfo("Initialized preprocessor node for " + camera_name)
    cam_id = camera_name[-1]
    pub = rospy.Publisher("preprocessed" + cam_id, PreprocessedMsg, queue_size=1)

    image_sub = rospy.Subscriber(
        camera_name + "/image_raw/compressed",
        CompressedImage,
        img_callback,
        callback_args=(processor, camera_name, pub),
        queue_size=1,
        buff_size=2 ** 24,
    )
    gps_sub = rospy.Subscriber("/fix", NavSatFix, gps_callback, queue_size=1)

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()


def img_callback(image, args):
    global CUR_GPS, send_frame, send_frame_raw

    processor = args[0]
    camera_name = args[1]
    pub = args[2]
    bridge = CvBridge()
    # frame = bridge.imgmsg_to_cv2(image, "rgb8")
    frame = bridge.compressed_imgmsg_to_cv2(
        image, desired_encoding="passthrough"
    )  # BGR images
    # frame = frame[:, :, ::-1]  # BGR to RGB
    x, y, w, h = 30, 30, 300, 60

    show_raw = False
    if show_raw:
        send_frame_raw += 1
        # Draw black background rectangle
        img1 = cv2.rectangle(frame, (x, x - 10), (x + w, y + h), (0, 0, 0), -1)
        # Add text
        cv2.putText(
            img1,
            "Frame No. " + str(send_frame_raw),
            (x + int(w / 10), y + int(h / 2)),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
        )
        cv2.namedWindow("Raw images", 0)
        cv2.imshow("Raw images", img1)
        cv2.waitKey(1)

    secs = image.header.stamp.secs
    nsecs = image.header.stamp.nsecs
    # time_stamps = "_{:0>10d}_{:0>9d}".format(secs, nsecs)
    timestamp_sec = float(secs + nsecs / 1e9)
    send_flag = processor.filtering(frame, timestamp_sec)

    if send_flag:
        send_frame += 1
        pub_data = PreprocessedMsg()
        pub_data.image_ts = image.header.stamp
        pub_data.image = image
        pub_data.lat = CUR_GPS.latitude
        pub_data.lon = CUR_GPS.longitude
        pub_data.alt = CUR_GPS.altitude
        pub.publish(pub_data)

        show_preprocessed = False
        if show_preprocessed:
            # Draw black background rectangle
            img2 = cv2.rectangle(frame, (x, x - 10), (x + w, y + h), (0, 0, 0), -1)
            # Add text
            cv2.putText(
                img2,
                "Frame No. " + str(send_frame),
                (x + int(w / 10), y + int(h / 2)),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
            )
            cv2.namedWindow("Preprocessed images", 0)
            cv2.imshow("Preprocessed images", img2)
            cv2.waitKey(1)

    # time.sleep(0.1)


def gps_callback(data):
    global LAST_GPS, CUR_GPS
    if data.status.status == -1:
        print("Unable to get a fix on the location.")
    else:
        LAST_GPS = CUR_GPS
        CUR_GPS = data


def main(args):
    preprocessor = Preprocessor(args)
    run_node(preprocessor, "camera" + str(args.cam_id))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocessing Node")
    # parser.add_argument("-i", "--input", default="./test.bag", help="Input ROS bag or folder")
    parser.add_argument(
        "-c",
        "--cam-id",
        type=int,
        default=5,
        help="Select camera ID to extract",
    )
    parser.add_argument("-o", "--output", default=None, help="Output dir")

    parser.add_argument(
        "--deduplicate",
        action="store_true",
        help="Whether to deduplicate the images",
    )
    parser.add_argument(
        "--del-blur",
        action="store_true",
        help="Whether to remove blurred images",
    )
    parser.add_argument(
        "--deduplicate-thres",
        type=int,
        default=15,
        help="Deduplcate threshold",
    )
    parser.add_argument(
        "--del-blur-thres",
        type=int,
        default=100,
        help="Deblur threshold",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=-1,
        help="Throttle at given frame rate",
    )
    parser.add_argument(
        "--gps-filtering",
        nargs=4,
        metavar=("Lat1", "Lon1", "Lat2", "Lon2"),
        type=float,
        default=None,
        help="Two GPS locations are needed for GPS filtering. \
            [Lat1, Lon1, Lat2, Lon2] in decimal degrees.\
            You could use GoogleMap to get the \
            upper-left and bottom-right corners)",
    )

    args = parser.parse_args()
    main(args)
