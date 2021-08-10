#!/bin/bash

# SPDX-FileCopyrightText: 2021 Carnegie Mellon University
#
# SPDX-License-Identifier: Apache-2.0

# Environments activate
source /opt/ros/melodic/setup.bash
source /home/albert/gabriel-BusEdge/client/ros_workspace/devel/setup.bash

GPS_LOG="/reg/v/gps/gps"
ACC_LOG="/reg/v/acceleration/acceleration"

while [ ! -f "$GPS_LOG" ] || [ ! -f "$ACC_LOG" ]
do
    echo "GPS/ACC logs are not created yet!"
    sleep 1 # or less like 0.2
done

# roslaunch --wait busedge_utils gps.launch &
# sleep 2
roslaunch --wait busedge_utils sensors.launch
sleep 1
exit 0
