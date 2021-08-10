# SPDX-FileCopyrightText: 2021 Carnegie Mellon University
#
# SPDX-License-Identifier: Apache-2.0

import os
import smtplib
from email import encoders
from email.header import Header
from email.mime.text import MIMEText
from email.utils import formataddr, parseaddr
from pathlib import Path

import rospy
from busedge_utils import get_disk_status


def _format_addr(s):
    name, addr = parseaddr(s)
    return formataddr((Header(name, "utf-8").encode(), addr))


def send(msg):

    # Environment variable should be set beforehand
    from_addr = os.getenv("BUS_EMAIL_FROM_ADDR")
    password = os.getenv("BUS_EMAIL_PASSWORD")
    to_addr = os.getenv("BUS_EMAIL_TO_ADDR")
    smtp_server = os.getenv("BUS_EMAIL_SMTP_SERVER")

    msg = MIMEText(msg, "plain", "utf-8")
    msg["From"] = _format_addr("BusEdge Client <%s>" % from_addr)
    msg["To"] = _format_addr("Admin <%s>" % to_addr)
    msg["Subject"] = Header("BusEdge Notification", "utf-8").encode()

    try:
        smtpObj = smtplib.SMTP_SSL(smtp_server, 465)
        smtpObj.login(from_addr, password)
        smtpObj.sendmail(from_addr, [to_addr], msg.as_string())
        rospy.loginfo("mail has been send successfully.")
        smtpObj.quit()
    except smtplib.SMTPException as e:
        rospy.logwarn("Exception happens.")
        rospy.logwarn(e)


def main():
    rospy.init_node("email_sender")
    msg = "Recorder Power on."

    disk_stat = get_disk_status.disk_stat("/media/albert/Elements/")
    record_directory = Path("/media/albert/Elements/RECORDS")
    rosbag_size = (
        sum(f.stat().st_size for f in record_directory.glob("**/*") if f.is_file())
        / 1024.0
        / 1024
        / 1024
    )
    strs = "\nStorage: avail - {:.2f} GB;    rosbag - {:.2f} GB\n".format(
        disk_stat["free"], rosbag_size
    )
    msg += strs
    send(msg)


if __name__ == "__main__":
    main()
