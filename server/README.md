<!--
SPDX-FileCopyrightText: 2021 Carnegie Mellon University

SPDX-License-Identifier: Apache-2.0
-->

# Bus-Edge Server

## Table of Contents

- [Code Structure](#code-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Add New Engines](#add-new-engines)

## Code Structure

- **engines**: Implementations of cognitive engines for different tasks.
  - **autodet\_engine**: Cognitive engine for the
    [AutoDetectron](engines/autodet_engine) application.
  - **common**: Engines for common usages.
  - **detector\_docker**: Object detector engine in a docker container.
  - **example\_engine\_docker**: Example cognitive engine in a docker container.
- **sinks**： Implementations of sinks for different tasks.
  - **livemap**: LiveMap web server.
  - **cvat**: Custermized CVAT client as the labeling tool.
- **protocol**： Protobuf files for common pipelines. See
  [Protocol Buffers][protocol_buffers] and [Gabriel Protocol][gabriel_protocol].
- **run\_server.py**: Main file to launch the gabriel server.

[protocol_buffers]: https://developers.google.com/protocol-buffers
[gabriel_protocol]: https://github.com/cmusatyalab/gabriel/blob/master/protocol/protos/gabriel.proto

## Installation

### Requirements

- Linux with Python >= 3.6
- Docker
- [Gabriel][gabriel]

### Install dependencies

- Install Gabriel related dependencies. See [Gabriel][gabriel] for more information.

    ```sh
    pip3 install gabriel-server\
        gabriel_protocol\
        imageio\
        opencv-python \
        protobuf \
        py-cpuinfo \
        PyQt5 \
        'pyzmq==18.1' \
        'websockets==8.0' \
        zmq
    ```

- Install docker. Each cogntive engine could run in its own docker container so
  that we could easily deploy a new algorithm on the platform without
  dependencies conflicts.

    ```sh
    curl -fsSL https://get.docker.com -o get-docker.sh
    sudo sh get-docker.sh
    # Optional: To run Docker with root privileges:
    sudo groupadd docker
    sudo usermod -aG docker $USER
    newgrp docker
    ```

- Install [nvidia-docker](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker)

  ```sh
  curl https://get.docker.com | sh \
    && sudo systemctl --now enable docker
  dist=$(. /etc/os-release;echo $ID$VERSION_ID) \
    && curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | \
       sudo apt-key add - \
    && curl -s -L https://nvidia.github.io/nvidia-docker/$dist/nvidia-docker.list | \
       sudo tee /etc/apt/sources.list.d/nvidia-docker.list
  sudo apt-get update
  sudo apt-get install -y nvidia-docker2
  sudo systemctl restart docker
  ```

- Compile and install BusEdge protocol.

    ```sh
    sudo apt-get install -y protobuf-compiler
    python3 protoc_compile.py
    cd protocol
    pip3 install .
    ```

- (Optional) [LiveMap](sinks/livemap) web server for visualization.
- (Optional) [CVAT](sinks/cvat) for image annotation.

## Usage

### Launch Gabriel Server

```sh
python3 run_server.py
```

### Launch Cognitive Engines

1. Option 1: Run cognitive engines without Docker

   ```sh
   python3 engines/[engine_name]/engine_runner.py
   ```

2. Option 2: Run cognitive engines in Docker containers

   ```sh
   # Build the image of the engine
   cd gabriel-BusEdge/server/engines/[engine_name]
   docker build -t your_cogntive_engine .
   ```

   ```sh
   # This will run the engines_runner.py in the docker container.
   docker run --gpus all --rm -it engine python3 engine_runner.py
   ```

   ```sh
   # Instead we could also run the bash shell.
   docker run --gpus all --rm -it engine /bin/bash
   ```

3. NOTE:
   - When launching engines in containers, if we need to save images for the
     LiveMap web server, add
     `-v your_host_path/images:/your_container_path/images`, which will share
     this folder between the container and the host.
   - `--build-arg USER_ID=${UID}` is recommended when building the docker image
     if we want to modify the mounted host volume, otherwise permission errors
     might occur.

### Launch Sinks

1. See [cvat](sinks/cvat), [livemap](sinks/livemap) or create your own sink module.

## Add New Engines

1. Design the pipeline for your task:
    - What sensor data do you need?
    - What algorithms would you like to use for early-discard filter on the
      client and cognitive engine on the server?
    - What messages would you like to transmit between the client and the server?
    - What sinks would you like to adopt?
2. Have your cognitive engine algorithm running in a docker container. Copy
   your repository as a engine folder under `./engines/`.
3. Define your protobuf format for the transmitted messages and compile it. See
   [example protocol](engines/example_engine_docker/protocol). (You might have
   done this for client.)
4. Add some lines to your Dockerfile to install some extra packages related to
   gabriel. See [example Dockerfile](engines/example_engine_docker/Dockerfile).
   (You might have done this for client.)
5. Write a engine\_runner to connect your engine to the gabriel server. See
   [example engine_runner](engines/example_engine_docker/engine_runner.py).
6. Now your container of cognitive engine is ready. `--gpus all` is needed if
   you want to use all the GPUs.

    ```sh
    docker build -t your_cogntive_engine .
    docker run --gpus all your_cogntive_engine
    ```

[gabriel]: https://github.com/cmusatyalab/gabriel
