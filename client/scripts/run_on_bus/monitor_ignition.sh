#!/bin/bash

# SPDX-FileCopyrightText: 2021 Carnegie Mellon University
#
# SPDX-License-Identifier: Apache-2.0


signal=`cat /reg/v/shutdown/ignition_off`
# signal=`cat /home/albert/ignition_off`
while [[ $signal -ne "1" ]] ; do
    # echo "The ignition is on with value ${signal}."
    sleep 10
    signal=`cat /reg/v/shutdown/ignition_off`
#     signal=`cat /home/albert/ignition_off`
done

echo "The ignition is off with value ${signal}. Now starting to stop the data collection!"
echo "Stopping the rosbag record node..."
RECORDS_FOLDER="/media/albert/Elements/RECORDS"
if [ -d ${RECORDS_FOLDER} ] ; then
    echo "Directory ${RECORDS_FOLDER} exists."
else
    echo "ERROR: Directory ${RECORDS_FOLDER} does not exists."
fi

source /opt/ros/melodic/setup.bash
rosnode kill /rosbag_record_busedge
sleep 10
exit 0
