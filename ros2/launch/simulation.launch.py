import os
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


 
def generate_launch_description():
  ld = LaunchDescription()  

  pkg_gazebo_ros='kris_description'
  pkg_custom_terrain='custom_gazebo_world'

  start_gazebo_world = IncludeLaunchDescription(
    PythonLaunchDescriptionSource(os.path.join(pkg_gazebo_ros, 'launch', 'gazebo.launch.py')))
  
  start_gazebo_terrain = IncludeLaunchDescription(
    PythonLaunchDescriptionSource(os.path.join(pkg_custom_terrain, 'launch','launch_terrain_items.launch.py')))
  

  cmd_vel_publisher_cmd = Node(
    package='gail_navigation', 
    executable='publish_cmd_vel.py',
   )
  
  ld.add_action(start_gazebo_world)
  ld.add_action(start_gazebo_terrain)
  ld.add_action(cmd_vel_publisher_cmd)

  return ld