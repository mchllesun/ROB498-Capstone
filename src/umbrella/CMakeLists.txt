cmake_minimum_required(VERSION 3.8)
project(umbrella)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclpy REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)

# Generate service and message interfaces
rosidl_generate_interfaces(${PROJECT_NAME}
  "srv/SetPoseSource.srv"
  "msg/Detection.msg"
  "msg/DetectionArray.msg"
  DEPENDENCIES std_msgs
)

install(
  PROGRAMS scripts/odom_republisher.py
  scripts/comms_node.py
  scripts/mission_planner.py
  scripts/imx_camera_driver.py
  scripts/obj_tracker_node.py
  scripts/dot_tracker_node.py
  scripts/waypoint_generator.py
  scripts/circle_search_node.py
  DESTINATION lib/${PROJECT_NAME}/
)

install(
  DIRECTORY launch models
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
