<!--
SPDX-FileCopyrightText: 2021 Carnegie Mellon University

SPDX-License-Identifier: Apache-2.0
-->

# Bus-Edge Client

## Table of Contents

- [Code Structure](#code-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Add New Filters](#add-new-filters)

## Code Structure

- **nodes**: Implementations of ROS nodes for different tasks. (Early-discard filters.)
  - **autodet\_node**: Early discard filter node for the
    [AutoDetectron](../server/engines/autodet_engine) application.
  - **common**: Nodes for common usages.
  - **detector\_docker**: Object detector node for early discard filtering in a
    docker container.
  - **example\_node\_docker**: Example early discard filter node in a docker container.
- **ros\_workspace\/src**: ROS packages to manage sensors' inputs. (Sensor
  drivers and recorder)
  - **camera\_driver**: Camera driver package.
  - **gps\_driver**: GPS driver package.
  - **util\_nodes**: Utility packages including system launcher and monitor
    nodes.
- **protocol**： Protobuf files for common pipelines. See
  [Protocol Buffers][protocol_buffers] and the original
  [Gabriel Protocol][gabriel_protocol].
- **scripts**: Utility scripts.
  - **run\_on\_bus**: Scripts and systemd unit files for the deployment on the bus.
  - **use\_rosbag**: Scripts to use the recorded rosbag data.
- **run\_client.py**: Main file to launch the gabriel client.

[protocol_buffers]: https://developers.google.com/protocol-buffers
[gabriel_protocol]: https://github.com/cmusatyalab/gabriel/blob/master/protocol/protos/gabriel.proto

## Installation

### Requirements

- Linux with Python >= 3.6
- ROS Melodic
- Docker
- [Gabriel](https://github.com/cmusatyalab/gabriel)

### Install dependencies

- Install ROS related dependencies. Here we only install the needed packages of
  ROS. See [ROS website](http://wiki.ros.org/ROS/Installation) for more
  information.

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

- Install Gabriel related dependencies. See
  [Gabriel](https://github.com/cmusatyalab/gabriel) for more information.

    ```sh
    pip3 install gabriel-client\
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

- Install docker. Each early-discard filter could run in its own docker
  container so that we could easily deploy a new algorithm on the platform
  without dependencies conflicts.

    ```sh
    curl -fsSL https://get.docker.com -o get-docker.sh
    sudo sh get-docker.sh
    # Optional: To run Docker with root privileges:
    sudo groupadd docker
    sudo usermod -aG docker $USER
    newgrp docker
    ```

- Compile and install BusEdge protocol.

    ```sh
    sudo apt-get install -y protobuf-compiler
    python3 protoc_compile.py
    cd protocol
    pip3 install .
    ```

### Build ROS packages (For bus client)

- Build all the packages in ROS workspace

    ```sh
    cd ros_workspace
    catkin_make
    source devel/setup.bash
    ```

- (Optional) Install some useful ROS tools.

    ```sh
    sudo apt-get install ros-melodic-rqt-gui \
        ros-melodic-image-transport \
        ros-melodic-image-transport-plugins \
        ros-melodic-image-proc
    ```

## Usage

### Run Drivers for Different Sensors

1. Start ROS master by running `roscore`.
2. Launch all the sensors' drivers by running:
   `roslaunch busedge_utils sensors.launch`
3. NOTE:
   - You could customize your own configuration files and launch files for
     different cameras in `ros_workspace/src/camera_driver/config/cameraX.yaml`
     and `ros_workspace/src/camera_driver/launch/ipcamX.launch`.
   - For SafetyVision Recorder:
     - Because the GPS/Acc serial ports are occupied by the recorder's firmware,
       we only read the GPS/Acc results from the recorder's log files in
       `/reg/v/[sensor]/`. Both the GPS log file and the Accelleration log file
       are updated every 1.0 second. We could change this interval in the
       recorder's settings.
     - If we want to retrieve the raw sensor data, we need to close the firmware
       by running `/home/rr9000/rr9000_app/Application/scripts/close.sh` script
       twice and run the driver node by
       `rosrun nmea_navsat_driver nmea_serial_driver _port:=/dev/gps_node _baud:=9600`.
     - Camera Static IP: POEx => 182.168.1.10x

### Record ROSBAG

1. Start recording

    ```sh
    roslaunch busedge_utils start_record.launch
    ```

2. NOTE:
    - For the usage of the recorded rosbag data, see
      [How to use rosbag](./scripts/use_rosbag).

### Run Gabriel Client

1. Before running the gabriel client, we should first guarantee that the
   gabriel server is running on the cloudlet. See [Bus-Edge Server](../server).
2. Launch the client

    ```sh
    python3 run_client.py
    ```

3. If you have early-discard filters running in docker containers, build and
   run them as well. `--net=host` is necessary to enable the communication
   between the container and the host using ROS. For example:

    ```sh
    cd nodes/your_filter_node
    docker build -t your_filter_node .
    docker run --net=host --rm -it your_filter_node [COMMAND]
    ```

<!--
### Start System Monitor

1. Launch the monitor node

    ```sh
    roslaunch busedge_utils monitor_node
    ```
-->

### Deployment on the bus

We use the systemd service manager to run the client codes as a serie of
startup services. When deploying our codes on the bus computer, we need to
firstly create these services and enable them, so that the bus computer can
automatically run the codes at startup. All the above steps will be carried out
in the startup services. See [How to run on bus](scripts/run_on_bus).

## Add New Filters

1. Design the pipeline for your task:
    - What sensor data do you need?
    - What algorithms would you like to use for early-discard filter on the
      client and cognitive engine on the server?
    - What messages would you like to transmit between the client and the server?
    - What sinks would you like to adopt?
2. Have your early-discard filter algorithm running in a docker container. Copy
   your repository as a node folder under `./nodes/`.
3. Define your protobuf format for the transmitted messages and compile it. See
   [example protocol](nodes/example_node_docker/protocol).
4. Add some lines to your Dockerfile to install some extra packages related to
   ROS and Gabriel. See
   [example Dockerfile](nodes/example_node_docker/Dockerfile).
5. Write a simple ROS node to subsribe to sensors' input and publish results to
   the Gabriel client. See
   [example filter node](nodes/example_node_docker/filter_node.py). Also modify
   `run_client.py` to add your pipeline to gabriel.
6. Now your container of early-discard filter is ready. `--net=host` is
   necessary to enable the communication between the container and the host
   using ROS.

    ```sh
    docker build -t your_filter_node .
    docker run --net=host your_filter_node
    ```

7. You might also want to add a new cognitive engine on the server. See
   [add new engines](../server/README.md/#add-new-engines).
