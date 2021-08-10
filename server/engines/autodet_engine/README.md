<!--
SPDX-FileCopyrightText: 2021 Carnegie Mellon University

SPDX-License-Identifier: Apache-2.0
-->

# Auto-Detectron Pipeline

## Overview

Auto-Detectron is an application upon the BusEdge platform which enables users
to deploy and run an object detection task for a given target using the live
bus data. It integrates labeling and recursive learning to acquire object
detectors and then automatically launches and updates them on our platform for
continuous data filtering and analytics.

## Installation

### On the server

- Build the container of the cognitive engine:

    ```sh
    # Build a docker image for the engine
    cd server/engines/detector_docker
    docker build --build-arg USER_ID=${UID} -t autodet_engine .
    ```

- Install the sink for manual annotation, see [CVAT](../server/sinks/cvat)

### On the client

- Build the container of the cognitive engine:

    ```sh
    # Build a docker image for the node
    cd client/nodes/autodet_node/
    docker build --build-arg USER_ID=${UID} -t autodet_node .
    ```

## Launch the Pipeline

### On the Server

- Launch the Gabriel server in a terminal:

    ```sh
    cd server
    python3 run_server.py
    ```

- Launch the sink CVAT:

    ```sh
    cd server/sinks/cvat
    python3 cvat_client.py -t [target_name]
    ```

- Launch the Cognitive Engine inside a docker container in another terminal:

    ```sh
    cd server/engines/autodet_engine/
    docker run --gpus all -v "$PWD":/home/appuser/gabriel_server --rm -it \
        autodet_engine sh run_auto_det.sh [target_name]
    ```

### On the Client

- Launch ROS master by running `roscore` in a terminal.

- Then launch the Gabriel client in another terminal:

    ```sh
    cd client
    export GABRIEL_SERVER_IP_ADDRESS={your_server_address}
    python3 run_client.py --source-name autoDet_[target_name]
    ```

- Launch the Early Discard Filter in a docker container for this pipeline:

    ```sh
    cd client/nodes/autodet_node/
    docker run --net=host -v "$PWD":/home/appuser/client_node --rm -it \
        autodet_node python3 sys_eval_node.py -t [target_name] -i ./cloudy_downtown
    ```

## Bootstrapping and annotation

- Upload 10 shots of images containing the targets for bootstrapping. Save them
  in the unlabeled folder
  (server/engines/autodet_engine/autoDet_tasks/[target_name]/unlabeled)
- Open the labeling tool and then your can just keep labeling. The model will
  be updated and deployed automatically in the background.

    ```sh
    # Forward the port
    ssh -L localhost:8080:localhost:8080 [account@your_server_ip]
    # Open the CVAT UI in your browser
    localhost:8080
    ```
