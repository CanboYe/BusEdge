#!/bin/bash

# SPDX-FileCopyrightText: 2021 Carnegie Mellon University
#
# SPDX-License-Identifier: Apache-2.0

# Environments activate
source /opt/ros/melodic/setup.bash
source /home/albert/gabriel-BusEdge/client/ros_workspace/devel/setup.bash
roscore &
sleep 2
rosrun rqt_image_view rqt_image_view &

roslaunch --wait busedge_utils gps.launch &
sleep 2
roslaunch --wait busedge_utils sensors.launch
sleep 1

exit 0
