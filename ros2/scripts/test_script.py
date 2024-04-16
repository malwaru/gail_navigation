#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import  Joy
from geometry_msgs.msg import Twist,PoseStamped
from rclpy.logging import LoggingSeverity
import numpy as np
# from gail_navigation.gazebo_connection import GazeboConnection
from kris_envs.wrappers.gazebo_connection import GazeboConnection

class TestSpawn(Node):
    def __init__(self) -> None:
        super().__init__('tracked_command_publisher')
        self.subscriber_enable_btn = self.create_subscription(Joy, 
								'/joy',
								self.enable_btn_callback,
								10)

        self.subscriber_enable_btn  # prevent unused variable warning

        self.publisher_subgoal_pose = self.create_publisher(
                                                PoseStamped,
                                                '/subgoal_pose',                                                
                                                10)
       
        # Publisher to pubsish person depth
        self.enable_btns = [0,0,0,0,0]
        sub_goal_pose_msg = PoseStamped()
        sub_goal_pose_msg.header.frame_id = "base_link"
        sub_goal_pose_msg.pose.position.x = 1.0
        sub_goal_pose_msg.pose.position.y = 1.0
        sub_goal_pose_msg.pose.position.z = 0.0

        # Assign orientation
        sub_goal_pose_msg.pose.orientation.x = 0.0    
        sub_goal_pose_msg.pose.orientation.y = 0.0
        sub_goal_pose_msg.pose.orientation.z = 0.0
        sub_goal_pose_msg.pose.orientation.w = 0.0
        self.pose_stamped = sub_goal_pose_msg
        self.gazebo = GazeboConnection()

 
    def enable_btn_callback(self,msg):
        '''
        Receive postion of the tracked person 
        
        '''
        self.enable_btns=[msg.buttons[0],msg.buttons[1],msg.buttons[2],msg.buttons[3],msg.buttons[4]]
        self.publish_action(self.pose_stamped)
        # self.get_logger().info(f'Received enable button command {self.enable_btns}')

    def publish_action(self,pose_stamped):
        '''
        Published cmd_vel commands based on the leader pose 

        Params
        -------
        pose_stamped:   geometry_msgs.msgs.PointStamped
                        The mid point of the leader in 3d Space w.r.t base_link

        Returns
        --------
        None
        '''
        # self.get_logger().info(f'Enable button command {self.enable_btn}')
        if self.enable_btns[0]==1:
            self.get_logger().info('Publishing subgoal pose')
            self.publisher_subgoal_pose.publish(pose_stamped)                                
        elif self.enable_btns[1]==1:
            self.get_logger().info('Resetting simulation')
            self.gazebo.reset_sim()               
        elif self.enable_btns[2]==1:
            self.get_logger().info('Reset world')
            self.gazebo.reset_world()

        elif self.enable_btns[3]==1:
            self.get_logger().info('respawn robot')
            self.gazebo.spawn_robot()

        elif self.enable_btns[4]==1:
            self.get_logger().info('Kill robot')
            self.gazebo.delete_entity()
     
            
   



def main(args=None):
    rclpy.logging._root_logger.log(
        'Starting navigation command velocity publication ...',
        LoggingSeverity.INFO
    )
    rclpy.init(args=args)
    node = TestSpawn()
    rclpy.spin(node)
    # Destroy the node explicitly  
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
