cmake_minimum_required(VERSION 3.10.2)
project(lidar_localization)

SET(CMAKE_BUILD_TYPE "Release")
SET(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O4 -Wall")
add_compile_options(-std=c++14)
add_definitions(-std=c++14)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  geometry_msgs
  sensor_msgs
  nav_msgs
  pcl_ros
  tf
  eigen_conversions
  message_generation 
  std_srvs
)

add_service_files(
   FILES
   saveMap.srv
   optimizeMap.srv
   saveOdometry.srv
)

## Generate messages in the 'msg' folder
add_message_files(
    FILES
    # synced GNSS-odo measurement:
    PosVel.msg
)

generate_messages(
   DEPENDENCIES
   std_msgs
   geometry_msgs
   sensor_msgs
   nav_msgs
)

set(ALL_TARGET_LIBRARIES "")

include(cmake/glog.cmake)
include(cmake/YAML.cmake)
include(cmake/geographic.cmake)
include(cmake/PCL.cmake)
include(cmake/sophus.cmake)
include(cmake/g2o.cmake)
include(cmake/ceres.cmake)

include_directories(include ${catkin_INCLUDE_DIRS})
include(cmake/global_defination.cmake)
catkin_package()

file(GLOB_RECURSE ALL_SRCS "*.cpp")
file(GLOB_RECURSE NODE_SRCS "src/apps/*_node.cpp")
list(REMOVE_ITEM ALL_SRCS ${NODE_SRCS})

add_executable(data_pretreat_node src/apps/data_pretreat_node.cpp ${ALL_SRCS})
add_dependencies(data_pretreat_node ${catkin_EXPORTED_TARGETS})
target_link_libraries(data_pretreat_node ${catkin_LIBRARIES} ${ALL_TARGET_LIBRARIES})

add_executable(lio_matching_node src/apps/lio_matching_node.cpp ${ALL_SRCS})
add_dependencies(lio_matching_node ${catkin_EXPORTED_TARGETS})
target_link_libraries(lio_matching_node ${catkin_LIBRARIES} ${ALL_TARGET_LIBRARIES})

add_executable(back_end_node src/apps/back_end_node.cpp ${ALL_SRCS})
add_dependencies(back_end_node ${catkin_EXPORTED_TARGETS})
target_link_libraries(back_end_node ${catkin_LIBRARIES} ${ALL_TARGET_LIBRARIES})

