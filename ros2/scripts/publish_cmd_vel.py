#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import  Joy
from geometry_msgs.msg import Twist,PoseStamped
from rclpy.logging import LoggingSeverity
import numpy as np

class CmdVelPublisher(Node):
    def __init__(self) -> None:
        super().__init__('tracked_command_publisher')
        self.subscriber_subgoal_pose = self.create_subscription(
                                                PoseStamped,
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
        self.cmd_vel_linear_max=0.4
        self.cmd_vel_angular_max=0.5
    
 
    def subgoal_pose_callback(self,msg):
        '''
        Receive postion of the tracked person 
        
        '''
        self.publish_goal_pose(msg)

    def pid_generator(self,error_linear,error_angular):
        '''
        
        '''
        vel_linear=0.0
        vel_angular=0.0
        # If goal is behind the robot, the robot will turn on the spot
        if error_linear<0.00:            
            vel_linear=0.0
            #Allows to turn faster 
            vel_angular=-1.0*min(error_angular*self.pid_angualar[0],self.cmd_vel_angular_max)+0.5

        else:
            proportional_linear=error_linear*self.pid_linear[0]
            integral_linear=self.error_total_linear*self.pid_linear[1]
            #Ignore dt term for simplicity
            derivative_linear=(error_linear-self.error_previous_linear)*self.pid_linear[2]
            vel_linear=proportional_linear+integral_linear+derivative_linear
            self.error_previous_linear=error_linear

            proportional_angular=error_angular*self.pid_angualar[0]
            integral_angular=self.error_total_angular*self.pid_angualar[1]
            #Ignore dt term for simplicity
            derivative_angular=(error_angular-self.error_previous_angular)*self.pid_angualar[2]
            vel_angular=proportional_angular+integral_angular+derivative_angular
            self.error_previous_angular=error_angular

            vel_linear=min(vel_linear,self.cmd_vel_linear_max)#+self.error_total_linear*self.pid_linear[1]
            vel_angular=-1.0*min(error_angular*self.pid_angualar[0],self.cmd_vel_angular_max)#+self.error_total_angular*self.pid_angualar[1]

       

        return vel_linear,vel_angular

    def velocity_generator(self,error_linear,error_angular):
        '''
        Generate velocity depending on the error
        The robot moves linearly while turning if the goal is in the positive 
        direction
        It goal is in negative direction the robot turn on the spot until
        the goal is in positive direction 
        
        '''
        vel_linear=0.0
        vel_angular=0.0
        ## Check for sign instead
        if error_linear<0.00:            
            vel_linear=0.0
            #Allows to turn faster 
            vel_angular=-1.0*min(error_angular*self.pid_angualar[0],self.cmd_vel_angular_max)+0.5
            

        else:
            vel_linear=min(error_linear*self.pid_linear[0],self.cmd_vel_linear_max)#+self.error_total_linear*self.pid_linear[1]
            vel_angular=-1.0*min(error_angular*self.pid_angualar[0],self.cmd_vel_angular_max)#+self.error_total_angular*self.pid_angualar[1]

       

        return vel_linear,vel_angular
   

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

        dx=pose_stamped.pose.position.x
        #This isssue need to be fixed right now by trial an error i found the mid point 
        # in the camera frame has a value of 0.38 
        dy=pose_stamped.pose.position.y
     
        error_linear=dx
        self.error_total_linear+=error_linear
        error_angular=np.arctan2(dy,dx)
        self.error_total_angular+=error_angular

        vel_linear,vel_angular=self.velocity_generator(error_linear,error_angular)
        
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
