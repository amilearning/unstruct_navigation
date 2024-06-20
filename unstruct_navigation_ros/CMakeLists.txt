#                                                                               
# Copyright (c) 2022-2024, ETH Zurich, Matias Mattamala, Jonas Frey.
# All rights reserved. Licensed under the MIT license.
# See LICENSE file in the project root for details.
#                                                                               
cmake_minimum_required(VERSION 3.0.2)
project(unstruct_navigation_ros)

find_package(catkin REQUIRED COMPONENTS
  rospy
  nav_msgs
  sensor_msgs
  std_msgs
  grid_map_msgs
  wild_visual_navigation_msgs
  dynamic_reconfigure
)
catkin_python_setup()

generate_dynamic_reconfigure_options(
  cfg/dyn.cfg  
  #...
)

catkin_package(
  CATKIN_DEPENDS
)



catkin_install_python(PROGRAMS  scripts/unstruct_feature_extractor_node.py scripts/unstruct_planner_node.py
                      DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})
