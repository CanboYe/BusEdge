#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2020-2021 Carnegie Mellon University
#
# SPDX-License-Identifier: Apache-2.0

"""Execute during setup to compile .proto files"""

import subprocess
from pathlib import Path

try:
    from protoc import PROTOC_EXE
except ImportError:
    PROTOC_EXE = "protoc"


def build(*_setup_kwargs):
    """Run the protocol buffer compiler"""

    for path in Path(".").rglob("*/*.proto"):
        print("Compiling", path)
        subprocess.run(
            "{} --python_out=. {}".format(PROTOC_EXE, path.name),
            cwd=path.parent,
            shell=True,
            check=False,
        )


if __name__ == "__main__":
    build()
