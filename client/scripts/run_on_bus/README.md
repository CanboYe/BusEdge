<!--
SPDX-FileCopyrightText: 2021 Carnegie Mellon University

SPDX-License-Identifier: Apache-2.0
-->

# How to automatically run the client at startup of the bus computer

This folder includes scripts and instructions to automatically run the client
codes at startup of the bus computer.

## Systemd

We use the systemd service manager to run the client codes as a serie of
startup services. When we deploy our codes on the bus computer, we need to
firstly create these services and enable them, so that the bus computer can
automatically run the codes at startup. Instructions are as follows:

1. Modify the systemd unit files in the folder systemd\_service. You might need
   to change the directory of ExecStart or ExecStop if you use a different user
   name or a different installation path on the bus computer.
2. Add the application services to systemd. After we finished the modification
   of the systemd unit files, we need to copy them to the systemd configuration
   directory. We want to run the services via a user instead of the root.

    ```sh
    cp systemd_service/*.service ~/.config/systemd/user/
    systemctl --user daemon-reload
    ```

3. Then you could check the added services by `systemctl --user list-unit-files
   busedge*`. The next step is to enable all of these services.

    ```sh
    systemctl --user enable busedge_launch_all \
    busedge_run_roscore \
    busedge_gabriel_client \
    busedge_sensor_driver \
    busedge_record \
    busedge_monitor_ignition \
    wait_for_harddrive.service \
    wait_for_network.service
    ```

4. Now the client codes will run automatically at startup of the bus computer.
   Some useful commands to manage or monitor the systemd service are as
   follows.

    ```sh
    systemctl --user restart busedge_*
    systemctl --user status busedge_*
    systemctl --user stop busedge_*
    journalctl --user -b -u busedge_*
    ```
