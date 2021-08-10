#!/bin/bash

# SPDX-FileCopyrightText: 2021 Carnegie Mellon University
#
# SPDX-License-Identifier: CC0-1.0

set -e

# setup ros environment
source "/opt/ros/melodic/setup.bash"
exec "$@"
