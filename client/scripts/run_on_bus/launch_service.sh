#!/bin/bash

# SPDX-FileCopyrightText: 2021 Carnegie Mellon University
#
# SPDX-License-Identifier: Apache-2.0

systemctl --user enable busedge_launch_all.service
sleep 1
systemctl --user start busedge_launch_all.service
sleep 1
exit 0
