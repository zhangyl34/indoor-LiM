CMAKE_MINIMUM_REQUIRED(VERSION 3.16.3)
PROJECT(lio)

# debug mdoe.
SET(CMAKE_BUILD_TYPE "Debug")

SET(CMAKE_CXX_STANDARD 14)
# This makes C++14 a requirement,
# and prevents a "decay" to C++98 when the compiler does not support C++14.
SET(CMAKE_CXX_STANDARD_REQUIRED ON)
# This disables the use of compiler-specific extensions.
# For example, by default, CMake passes -std=gnu++14 to GCC on Linux.
# We want to build with -std=c++14. 
SET(CMAKE_CXX_EXTENSIONS OFF)

# Don't change the values of CMAKE_CXX_FLAGS,
# as those are mainly intended to be changed through toolchain files or the cache.
SET(CMAKE_CXX_FLAGS "-std=c++14 -pthread -O3")  # ikd-Tree

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
FIND_PATH(LOGGER_INCLUDE_DIR NAMES file_logger.h PATHS /home/neal/usr/include/file_logger)
FIND_LIBRARY(LOGGER_LIBRARIES file_logger HINTS /home/neal/usr/lib)

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
  ${LOGGER_INCLUDE_DIR}
)

ADD_EXECUTABLE(lio_node src/laserMapping.cpp src/preprocess.cpp src/IMU_Processing.cpp src/use-ikfom.cpp src/common_lib.cpp)# include/ikd-Tree/ikd_Tree.cpp)

TARGET_LINK_LIBRARIES(lio_node ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${IKD_LIBRARIES} ${LOGGER_LIBRARIES})
