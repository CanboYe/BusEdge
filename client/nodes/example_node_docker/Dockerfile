# SPDX-FileCopyrightText: 2021 Carnegie Mellon University
#
# SPDX-License-Identifier: Apache-2.0

FROM ubuntu:18.04
#-----------------#
# Your Dockerfile #
#-----------------#

# Install ROS related packages
RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
RUN apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654
RUN apt-get update
RUN apt-get install -y ros-melodic-ros-base ros-melodic-rosbag python3-numpy python3-yaml ros-melodic-cv-bridge
RUN pip3 install pycryptodomex gnupg rospkg

# Install gabriel related packages
RUN pip3 install gabriel-client\
    gabriel_protocol\
    imageio\
    opencv-python \
    protobuf \
    py-cpuinfo \
    PyQt5 \
    'pyzmq==18.1' \
    'websockets==8.0' \
    zmq


#-----------------#
# Your Dockerfile #
#-----------------#

ENTRYPOINT ["bash", "./ros_entrypoint.sh"]
CMD [ "python3", "./filter_node.py" ]
