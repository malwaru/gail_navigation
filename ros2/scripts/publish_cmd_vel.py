#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import  Joy
from geometry_msgs.msg import Twist,PointStamped
from rclpy.logging import LoggingSeverity
import numpy as np

class CmdVelPublisher(Node):
    def __init__(self) -> None:
        super().__init__('tracked_command_publisher')
        self.subscriber_subgoal_pose = self.create_subscription(
                                                PointStamped,
                                                '/subgoal_pose',
                                                self.subgoal_pose_callback,
                                                10)
        self.subscriber_subgoal_pose  # prevent unused variable warning
       
        # Publisher to pubsish person depth
        self.publisher_command_vel = self.create_publisher(
                                                Twist,
                                                '/cmd_vel', 
                                                10)
        self.target_tole=1.0
        #Integral of error
        self.error_total_linear=0.0
        self.error_total_angular=0.0
        #For differential of error
        self.error_previous_linear=0.0
        self.error_previous_angular=0.0
        #PID values
        # The array are in the order [P I D]
        self.pid_linear=[0.6,0.8,0.4]
        self.pid_angualar=[0.8,0.8,0.4]
        #Maximum allowable velocities
        self.cmd_vel_linear_max=0.8
        self.cmd_vel_angular_max=1.0
    
 
    def subgoal_pose_callback(self,msg):
        '''
        Receive postion of the tracked person 
        
        '''
        self.publish_goal_pose(msg)

    def publish_goal_pose(self,pose_stamped):
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

        dx=pose_stamped.point.x
        #This isssue need to be fixed right now by trial an error i found the mid point 
        # in the camera frame has a value of 0.38 
        dy=pose_stamped.point.y
     
        error_linear=dx
        self.error_total_linear+=error_linear
        error_angular=np.arctan2(dy,dx)
        self.error_total_angular+=error_angular

        
        vel_linear=0.0
        vel_angular=0.0

        vel_linear=min(error_linear*self.pid_linear[0],self.cmd_vel_linear_max)#+self.error_total_linear*self.pid_linear[1]
        vel_angular=-1.0*min(error_angular*self.pid_angualar[0],self.cmd_vel_angular_max)#+self.error_total_angular*self.pid_angualar[1]

        #Assuming the robot does not reverse
        if error_linear<0.0:
            vel_angular=0.0
            vel_linear=0.0
   
        velocity=Twist()
        velocity.linear.x=vel_linear
        velocity.angular.z=-vel_angular
        self.publisher_command_vel.publish(velocity)
            
   



def main(args=None):
    rclpy.logging._root_logger.log(
        'Starting navigation command velocity publication ...',
        LoggingSeverity.INFO
    )
    rclpy.init(args=args)
    node = CmdVelPublisher()
    rclpy.spin(node)
    # Destroy the node explicitly  
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
