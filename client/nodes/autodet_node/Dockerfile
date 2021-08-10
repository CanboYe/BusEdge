# SPDX-FileCopyrightText: 2021 Carnegie Mellon University
#
# SPDX-License-Identifier: Apache-2.0

# SPDX-FileCopyrightText: 2021 Carnegie Mellon University
#
# SPDX-License-Identifier: Apache-2.0

FROM ubuntu:bionic

ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && \
    apt-get install --no-install-recommends -y python3-pip python3.8-dev && \
    apt-get install -y \
    vim libpq-dev postgresql-client postgresql python3-pyqt5\
	python3-opencv ca-certificates python3-dev git wget sudo ninja-build gnupg curl

# Install ROS related packages
RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu bionic main" > /etc/apt/sources.list.d/ros-latest.list'
RUN apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654
RUN apt-get update
RUN apt-get install -y ros-melodic-ros-base ros-melodic-rosbag python3-numpy python3-yaml ros-melodic-cv-bridge

# RUN ln -s /usr/bin/pip3 /usr/bin/pip
# RUN ln -s /usr/bin/python3.8 /usr/bin/python

# create a non-root user
ARG USER_ID=1000
RUN useradd -m --no-log-init --system  --uid ${USER_ID} appuser -g sudo
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
USER appuser
WORKDIR /home/appuser

ENV PATH="/home/appuser/.local/bin:${PATH}"
RUN wget https://bootstrap.pypa.io/get-pip.py && \
	python3 get-pip.py --user && \
	rm get-pip.py

RUN pip install --user pycryptodomex gnupg rospkg

# install dependencies
# See https://pytorch.org/ for other options
RUN pip install --user tensorboard cmake   # cmake from apt-get is too old
RUN pip install --user torch==1.9.0+cpu torchvision==0.10.0+cpu torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html

RUN pip install --user 'git+https://github.com/facebookresearch/fvcore'

# install detectron2
RUN pip install --user detectron2 -f \
  https://dl.fbaipublicfiles.com/detectron2/wheels/cpu/torch1.9/index.html
# RUN git clone https://github.com/facebookresearch/detectron2 detectron2_repo
# RUN pip install --user -e detectron2_repo

# Set a fixed model cache directory.
ENV FVCORE_CACHE="/home/appuser/client_node/models"

# install D2Go
RUN pip install --user 'git+https://github.com/facebookresearch/mobile-vision.git'
RUN git clone https://github.com/facebookresearch/d2go d2go_repo
RUN pip install --user -e d2go_repo

# Set a var to indicate running in docker container
ENV AM_I_IN_A_DOCKER_CONTAINER Yes

# self-defined commands
RUN pip install --user gabriel-client\
    gabriel_protocol\
    imageio\
    opencv-python \
    protobuf \
    py-cpuinfo \
    PyQt5 \
    'pyzmq==18.1' \
    'websockets==8.0' \
    zmq \
    scikit-learn \
    tqdm
RUN mkdir -p /home/appuser/client_node
WORKDIR /home/appuser/client_node
COPY . /home/appuser/client_node

ENTRYPOINT ["bash", "./ros_entrypoint.sh"]
# CMD [ "python3", "./filter_node.py" ]
