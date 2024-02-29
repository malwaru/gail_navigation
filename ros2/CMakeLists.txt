cmake_minimum_required(VERSION 3.8)
project(gail_navigation)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclpy REQUIRED)
find_package(std_msgs REQUIRED)
find_package(rosbag2_cpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(rosgraph_msgs REQUIRED)
find_package(diagnostic_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(OpenCV 4 REQUIRED)




install(DIRECTORY
  config launch
  DESTINATION share/${PROJECT_NAME}
)

# Include Cpp "include" directory
include_directories(include)

# add_executable(ros_bag_player src/ros_bag_player.cpp)
# ament_target_dependencies(ros_bag_player rclcpp rosbag2_cpp std_msgs 
#                           geometry_msgs sensor_msgs nav_msgs rosgraph_msgs
#                           diagnostic_msgs) 
# Install Cpp executables
# install(TARGETS
# ros_bag_player
#   DESTINATION lib/${PROJECT_NAME}
# )


# Install Python modules
ament_python_install_package(${PROJECT_NAME})
# Install Python executables
install(PROGRAMS
  scripts/database_creator.py
  # scripts/kris_envs/envs/gazebo_connection.py
  DESTINATION lib/${PROJECT_NAME}
)


# if(BUILD_TESTING)
#   find_package(ament_lint_auto REQUIRED)
#   # the following line skips the linter which checks for copyrights
#   # comment the line when a copyright and license is added to all source files
#   set(ament_cmake_copyright_FOUND TRUE)
#   # the following line skips cpplint (only works in a git repo)
#   # comment the line when this package is in a git repo and when
#   # a copyright and license is added to all source files
#   set(ament_cmake_cpplint_FOUND TRUE)
#   ament_lint_auto_find_test_dependencies()
# endif()

ament_package()
