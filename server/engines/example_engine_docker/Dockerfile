# SPDX-FileCopyrightText: 2021 Carnegie Mellon University
#
# SPDX-License-Identifier: Apache-2.0

FROM ubuntu:18.04

#-----------------#
# Your Dockerfile #
#-----------------#

# Set a var to indicate running in docker container
ENV AM_I_IN_A_DOCKER_CONTAINER Yes

# Install gabriel related packages
RUN pip3 install gabriel-server\
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
