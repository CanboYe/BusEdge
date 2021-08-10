# SPDX-FileCopyrightText: 2021 Carnegie Mellon University
#
# SPDX-License-Identifier: Apache-2.0

FROM tensorflow/tensorflow:2.4.0

ARG DEBIAN_FRONTEND=noninteractive

# Install apt dependencies
RUN apt-get update && apt-get install -y \
    git \
    gpg-agent \
    python3-cairocffi \
    protobuf-compiler \
    python3-pil \
    python3-lxml \
    python3-tk \
    wget \
    lsb-release \
    vim

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

# See https://stackoverflow.com/questions/65405705/undefined-symbol-zn10tensorflow8opkernel11tracestringepns-15opkernelcontexteb
RUN rm /usr/local/lib/python3.6/dist-packages/tensorflow/core/kernels/libtfkernel_sobol_op.so

# create a non-root user without password
ARG USER_ID=1000
RUN useradd -m --no-log-init --system  --uid ${USER_ID} appuser -g sudo
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
USER appuser
WORKDIR /home/appuser


RUN mkdir -p /home/appuser/tensorflow/models
RUN git clone https://github.com/tensorflow/models.git /home/appuser/tensorflow/models


# Compile protobuf configs
WORKDIR /home/appuser/tensorflow/models/research
RUN protoc object_detection/protos/*.proto --python_out=.
RUN cp object_detection/packages/tf2/setup.py ./


RUN python3 -m pip install -U pip
RUN python3 -m pip install .

ENV TF_CPP_MIN_LOG_LEVEL 3
ENV PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim

# Will need to set the ROS environment variable by: bash /opt/ros/melodic/setup.bash before running the codes
# Or set an entrypoint like: https://github.com/osrf/docker_images/blob/master/ros/noetic/ubuntu/focal/ros-core/ros_entrypoint.sh

RUN mkdir -p /home/appuser/client_node
WORKDIR /home/appuser/client_node
COPY . /home/appuser/client_node

ENTRYPOINT ["bash", "./ros_entrypoint.sh"]
CMD [ "python3", "./sign_filter_node.py" ]
