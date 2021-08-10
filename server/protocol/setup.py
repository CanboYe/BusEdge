# SPDX-FileCopyrightText: 2021 Carnegie Mellon University
#
# SPDX-License-Identifier: Apache-2.0

import setuptools

DESCRIPTION = "Common Protocol for the BusEdge Platform"

setuptools.setup(
    name="busedge-protocol",
    version="1.0",
    author="Canbo Ye",
    author_email="albert.yip@hotmail.com",
    description=DESCRIPTION,
    url="https://github.com/CanboYe/gabriel-BusEdge",
    packages=["busedge_protocol"],
    license="Apache-2.0",
    classifiers=[
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.5",
    install_requires=[
        "protobuf>=3.12",
    ],
)
