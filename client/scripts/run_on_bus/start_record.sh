#!/bin/bash

# SPDX-FileCopyrightText: 2021 Carnegie Mellon University
#
# SPDX-License-Identifier: Apache-2.0

# Environments activate
source /opt/ros/melodic/setup.bash
source /home/albert/gabriel-BusEdge/client/ros_workspace/devel/setup.bash

sleep 3
# RECORDS_FOLDER="/home/albert/RECORDS"
RECORDS_FOLDER="/media/albert/Elements/RECORDS"

((count = 1000))                            # Maximum number to try.
while [[ $count -ne 0 ]] ; do
    if [ -d ${RECORDS_FOLDER} ] ; then
        echo "Directory ${RECORDS_FOLDER} exists."
        ((count = 1))                      # If okay, flag to exit loop.
    else
        echo "Directory ${RECORDS_FOLDER} does not exists. Retrying"
        sleep 2
    fi
    ((count = count - 1))                  # So we don't go forever.
done

RECORDS_DIR="${RECORDS_FOLDER}/$(date +%Y_%m_%d_%H_%M)"
mkdir -p "${RECORDS_DIR}"
cd "${RECORDS_FOLDER}"

FREE=`df -k --output=avail "$PWD" | tail -n1`   # df -k not df -h
echo "Free space = $(($FREE/1024/1024)) GBs"
if [[ $FREE -lt 314572800 ]]; then               # 10G = 10*1024*1024k
    echo "Less than 300 GBs free!"
    echo "Removing the oldest records"
    ls -1t | tail -n 1 | xargs rm -rf
fi;

rosbag record -o "${RECORDS_DIR}/bus" -b 10240 --split --size=5120 --max-splits 60 -e '/camera(.*)/image_raw/compressed|/camera(.*)/camera_info|/fix|/acceleration|/velocity' __name:=rosbag_record_busedge
# roslaunch busedge_utils start_record_test.launch &
sleep 1
exit 0
