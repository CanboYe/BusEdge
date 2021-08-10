#!/bin/bash

# SPDX-FileCopyrightText: 2021 Carnegie Mellon University
#
# SPDX-License-Identifier: Apache-2.0

((count = 1000))                            # Maximum number to try.
while [[ $count -ne 0 ]] ; do
    ping -c 1 8.8.8.8                      # Try once.
    rc=$?
    if [[ $rc -eq 0 ]] ; then
        ((count = 1))                      # If okay, flag to exit loop.
    else
        sleep 2
    fi
    ((count = count - 1))                  # So we don't go forever.
done

if [[ $rc -eq 0 ]] ; then                  # Make final determination.
    echo "The internet is up."
else
    echo "Network Connection Timeout."
    exit 0
fi

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

# export BUS_EMAIL_FROM_ADDR=
# export BUS_EMAIL_PASSWORD=
# export BUS_EMAIL_TO_ADDR=
# export BUS_EMAIL_SMTP_SERVER=
# export GABRIEL_SERVER_IP_ADDRESS=

source /opt/ros/melodic/setup.bash
source /home/albert/gabriel-BusEdge/client/ros_workspace/devel/setup.bash
sleep 1

rosrun busedge_utils email_sender &
# rosrun rqt_gui rqt_gui &

# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/home/albert/anaconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/home/albert/anaconda3/etc/profile.d/conda.sh" ]; then
        . "/home/albert/anaconda3/etc/profile.d/conda.sh"
    else
        export PATH="/home/albert/anaconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<

# Environments activate
# Add Tensorflow Libraries to PYTHONPATH
export PYTHONPATH=$PYTHONPATH:/home/albert/workspace/tensorflow/models/research:/home/albert/workspace/tensorflow/models/research/slim

jupyter notebook &

WORKSPACE="/home/albert/gabriel-BusEdge/client"
cd "${WORKSPACE}"
# For visualization
export QT_X11_NO_MITSHM=1
# Run client
# docker run --net=host --rm detector python3 sign_filter_node.py &
python3 run_client.py 2> /home/albert/LOG/client_$(date +%Y-%m-%d_%H:%M:%S).log

# python3 run_client.py -v -r -c 1 2 3 4 5 2> /home/albert/LOG/client_$(date +%Y-%m-%d_%H:%M:%S).log

sleep 1
exit 0
