CMAKE_MINIMUM_REQUIRED(VERSION 3.16.3)
PROJECT(lio)

SET(CMAKE_BUILD_TYPE "Debug")

SET(CMAKE_CXX_STANDARD 14)
SET(CMAKE_CXX_STANDARD_REQUIRED ON)
SET(CMAKE_CXX_EXTENSIONS OFF)

ADD_DEFINITIONS(-DROOT_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/\")

FIND_PACKAGE(catkin REQUIRED COMPONENTS
  geometry_msgs
  nav_msgs
  pcl_ros
  roscpp
  sensor_msgs
  tf
  livox_ros_driver
  message_generation
)

FIND_PACKAGE(Eigen3 REQUIRED)

FIND_PACKAGE(PCL 1.8 REQUIRED)

FIND_PATH(IKD_INCLUDE_DIR NAMES ikd_Tree.h PATHS /home/neal/usr/include/ikd_Tree)
FIND_LIBRARY(IKD_LIBRARIES ikd_Tree HINTS /home/neal/usr/lib)

ADD_MESSAGE_FILES(
  FILES
  Pose6D.msg
)

GENERATE_MESSAGES(
  DEPENDENCIES
  geometry_msgs
)

CATKIN_PACKAGE(
  INCLUDE_DIRS include
  CATKIN_DEPENDS geometry_msgs nav_msgs pcl_ros roscpp sensor_msgs tf livox_ros_driver message_generation
  DEPENDS EIGEN3 PCL
)

INCLUDE_DIRECTORIES(
  include
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
  ${PCL_INCLUDE_DIRS}
  ${IKD_INCLUDE_DIR}
)

ADD_EXECUTABLE(lio_node src/laserMapping.cpp src/preprocess.cpp src/IMU_Processing.cpp src/use-ikfom.cpp src/common_lib.cpp)

TARGET_LINK_LIBRARIES(lio_node ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${IKD_LIBRARIES})
