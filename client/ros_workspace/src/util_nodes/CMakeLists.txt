# SPDX-FileCopyrightText: 2021 Carnegie Mellon University
#
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 2.8.3)
project(busedge_utils)

find_package(catkin REQUIRED
            roscpp
            rospy
            std_msgs
            sensor_msgs
            message_generation)

catkin_python_setup()

add_message_files(
  FILES
  ErrMsg.msg
  PreprocessedMsg.msg
  FeatureMsg.msg
)

generate_messages(
  DEPENDENCIES
  std_msgs
  sensor_msgs
)

catkin_package(CATKIN_DEPENDS message_runtime)

install(PROGRAMS
   scripts/monitor_node
   DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch/
   DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
   FILES_MATCHING PATTERN "*.launch"
)

if (CATKIN_ENABLE_TESTING)
  find_package(roslint)
  roslint_python()
  roslint_add_test()
endif()
