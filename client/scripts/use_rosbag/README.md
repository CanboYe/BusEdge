<!--
SPDX-FileCopyrightText: 2021 Carnegie Mellon University

SPDX-License-Identifier: Apache-2.0
-->

# How to use the collected rosbag

This folder includes scripts and instructions to use the collected rosbag data.

## Common usages

1. Download rosbag from our data center under `/Data/busedge`. You can also download
   a sample rosbag from this
   [link](https://drive.google.com/drive/folders/1kO9c3BQtAWeBVQN8p3Yhq0NuJscrG5Uf?usp=sharing).
2. Installation of ROS related packages

    ```sh
    sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu \
        $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
    sudo apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' \
        --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654
    sudo apt-get update
    sudo apt-get install -y ros-melodic-ros-base ros-melodic-rosbag \
        python3-numpy python3-yaml ros-melodic-cv-bridge
    echo "source /opt/ros/melodic/setup.bash" >> ~/.bashrc
    source ~/.bashrc
    python3 -m pip install opencv-python pycryptodomex gnupg rospkg piexif
    ```

3. Method 1: Extract data from a rosbag directly and process it in an offline manner.

    ```sh
    python3 read_bag.py -i path_to_bag -o path_to_output --save-gps --cam-id 3
    ```

4. Method 2: Play a rosbag and subscribe to the topics. Process the data in an
   online manner (as if on the bus).

    ```sh
    roscore    # in one terminal
    rosbag play your_bag_name.bag    # in another terminal
    python3 subscribe_bag.py -o path_to_output --save-gps --cam-id 3  # in another terminal
    ```

5. Notice: These two scripts only extract images from rosbag and save them into a folder.
   Add --save-gps if you want to save GPS as exif info of the jpeg images. Use --cam-id
   to specify the camera you want to look at. We have data of all the 5 cameras inside
   the rosbag.

## Useful tools

1. If you are processing the rosbag data in an online manner, you might find
   [rostopic](http://wiki.ros.org/rostopic) and
   [rqt\_gui](http://wiki.ros.org/rqt_image_view) useful. For example:

    ```sh
    rostopic list
    rostopic echo your_topic_name
    rosrun rqt_image_view rqt_image_view
    ```

2. If undistorted images are needed, run image\_proc and subscribe to
   [/camera3/image\_rect\_color](http://wiki.ros.org/image_proc). You will
   need to repulish compressed images to raw images first if using the
   collected rosbag.

    ```sh
    rosrun image_transport republish compressed in:=/camera3/image_raw raw out:=/camera3/image_raw
    ROS_NAMESPACE=camera3 rosrun image_proc image_proc
    ```

3. If you want to throttle image messages, use
   [topic\_tools/throttle](http://wiki.ros.org/topic_tools/throttle). You could
   also do it in the scripts provided.

    ```sh
    rosrun topic_tools throttle messages /camera3/image_raw 1.0
    ```

4. If you want to trim a rosbag into a smaller with given topics or timestamps,
   you can use `rosbag filter`. For example:

    ```sh
    rosbag filter input.bag output.bag \
        "t.secs >= 1612824008 and t.secs <= 1612824038 and topic == '/fix'"
    ```

5. List of recorded ROS topics:
    - /camera\*/camera\_info: camera params
    - /camera\*/image\_raw/compressed: compressed jpeg images
    - /fix: GPS data
    - /acceleration
    - /velocity
