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
        sleep 1
    fi
    ((count = count - 1))                  # So we don't go forever.
done
exit 0
