<!--
SPDX-FileCopyrightText: 2021 Carnegie Mellon University

SPDX-License-Identifier: Apache-2.0
-->

# CVAT sink

## Install CVAT

1. See this [repo](https://github.com/openvinotoolkit/cvat/blob/develop/site/content/en/docs/for-users/installation.md).
2. Set environment variable:

    ```sh
    export CVAT_USER={CVAT-USERNAME}
    export CVAT_PASS={CVAT-PASSWORD}
    ```

## Launch CVAT

1. Customize your configurations in `./cfg` folder.
2. Launch CVAT client with `python3 cvat_client.py`.
