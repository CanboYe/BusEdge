#!/bin/bash

# SPDX-FileCopyrightText: 2021 Carnegie Mellon University
#
# SPDX-License-Identifier: Apache-2.0

RECORDS_FOLDER="/media/albert/Elements/RECORDS"
((count = 10000))                            # Maximum number to try.
while [[ $count -ne 0 ]] ; do
    if [ -d ${RECORDS_FOLDER} ] ; then
        echo "Directory ${RECORDS_FOLDER} exists."
        ((count = 1))                      # If okay, flag to exit loop.
    else
        echo "Directory ${RECORDS_FOLDER} does not exists. Retrying"
        sleep 1
    fi
    ((count = count - 1))                  # So we don't go forever.
done
exit 0
