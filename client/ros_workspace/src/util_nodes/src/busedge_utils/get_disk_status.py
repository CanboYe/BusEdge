# SPDX-FileCopyrightText: 2021 Carnegie Mellon University
#
# SPDX-License-Identifier: Apache-2.0

import os


def disk_stat(folder):
    """
    check disk status
    :param folder dir
    :return:
    """
    hd = {}
    disk = os.statvfs(folder)
    hd["free"] = disk.f_bavail * disk.f_frsize / (1024 * 1024 * 1024.0)
    hd["total"] = disk.f_blocks * disk.f_frsize / (1024 * 1024 * 1024.0)
    hd["used"] = hd["total"] - hd["free"]
    hd["used_proportion"] = float(hd["used"]) / float(hd["total"])
    return hd


# print(disk_stat('.'))
