#!/bin/bash

# SPDX-FileCopyrightText: 2021 Carnegie Mellon University
#
# SPDX-License-Identifier: Apache-2.0

echo "Stopping the rosbag record node..."

RECORDS_FOLDER="/media/albert/Elements/RECORDS"

if [ -d ${RECORDS_FOLDER} ] ; then
    echo "Directory ${RECORDS_FOLDER} exists."
else
    echo "ERROR: Directory ${RECORDS_FOLDER} does not exists."
fi

source /opt/ros/melodic/setup.bash
rosnode kill /rosbag_record_busedge
sleep 3
exit 0
