# SPDX-FileCopyrightText: 2021 Carnegie Mellon University
#
# SPDX-License-Identifier: Apache-2.0

FROM nvidia/cuda:11.1.1-cudnn8-devel

ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && apt-get install -y \
    vim libpq-dev postgresql-client postgresql python3-pyqt5\
	python3-opencv ca-certificates python3-dev git wget sudo ninja-build
RUN ln -sv /usr/bin/python3 /usr/bin/python

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

# install dependencies
# See https://pytorch.org/ for other options if you use a different version of CUDA
RUN pip install --user tensorboard cmake   # cmake from apt-get is too old
RUN pip install --user torch==1.9 torchvision==0.10 -f https://download.pytorch.org/whl/cu111/torch_stable.html

RUN pip install --user 'git+https://github.com/facebookresearch/fvcore'
# install detectron2
RUN git clone https://github.com/facebookresearch/detectron2 detectron2_repo
# set FORCE_CUDA because during `docker build` cuda is not accessible
ENV FORCE_CUDA="1"
# This will by default build detectron2 for all common cuda architectures and take a lot more time,
# because inside `docker build`, there is no way to tell which architecture will be used.
ARG TORCH_CUDA_ARCH_LIST="Kepler;Kepler+Tesla;Maxwell;Maxwell+Tegra;Pascal;Volta;Turing"
ENV TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}"

RUN pip install --user -e detectron2_repo

# Set a fixed model cache directory.
ENV FVCORE_CACHE="/home/appuser/gabriel_server/models"
# WORKDIR /home/appuser/detectron2_repo

# install D2Go
RUN pip install --user 'git+https://github.com/facebookresearch/mobile-vision.git'
RUN git clone https://github.com/facebookresearch/d2go d2go_repo
RUN pip install --user -e d2go_repo

# Set a var to indicate running in docker container
ENV AM_I_IN_A_DOCKER_CONTAINER Yes

# self-defined commands
RUN pip install --user gabriel-server\
    gabriel_protocol\
    imageio\
    opencv-python \
    protobuf \
    py-cpuinfo \
    PyQt5 \
    'pyzmq==18.1' \
    'websockets==8.0' \
    zmq \
    psycopg2 \
    piexif \
    scikit-learn \
    tqdm
RUN mkdir -p /home/appuser/gabriel_server
WORKDIR /home/appuser/gabriel_server
COPY . /home/appuser/gabriel_server
# RUN python3 engines_runner.py
