#!/usr/bin/env bash

# SPDX-FileCopyrightText: 2021 Carnegie Mellon University
#
# SPDX-License-Identifier: Apache-2.0

if [ "$#" -ne "1" ]
then
    echo "Usage: sh run_auto_det.sh [target_name]"
    exit 0
else
    target_name=$1
    echo "Running auto-detectron with target_name $target_name"
    echo "The source name for this new created pipeline is autoDet_$target_name"
    echo "All the results and models will be save in ./autoDet_tasks/$target_name"
fi

# add CUDA_VISIBLE_DEVICES=[gpu_id] if want to use different gpu.
# add --use-svm to the cloudlet script if want to use svm
python model_manager_bus.py --target-name $target_name &
pid_0=$!
python model_manager_cloudlet.py --use-svm --target-name $target_name &
pid_1=$!
python anno_manager.py --target-name $target_name &
pid_2=$!
python engine_runner.py --use-svm --target-name $target_name &
pid_3=$!
trap "kill ${pid_0} ${pid_1} ${pid_2} ${pid_3}; exit 1" INT
wait
