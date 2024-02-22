#!/usr/bin/env python3
from typing import Any
import gymnasium as gym
from gymnasium import spaces
import rclpy
import rclpy.node as Node
from sensor_msgs.msg import Image,CameraInfo
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist,PoseStamped
from cv_bridge import CvBridge
import cv2
import numpy as np
from GailNavigationNetwork.model import NaviNet



class KrisEnv(gym.Env,Node):
    def __init__(self):
        super().__init__('kris_env_node')

        #ROS initializations
        self.image_dim=(240,320)          
        self.depth_image_raw_sub = self.create_subscription(
            Image, '/framos/depth/image_raw', self.depth_image_raw_callback, 10)
        self.image_raw_sub = self.create_subscription(
            Image, '/framos/image_raw', self.image_raw_callback, 10)
        self.odometry_filtered_sub = self.create_subscription(
            Odometry, '/odometry/filtered', self.odometry_filtered_callback, 10)
        self.goal_pose_sub = self.create_subscription(
            PoseStamped, '/goal_pose', self.goal_pose_callback, 10)
        self.camera_info_sub = self.create_subscription(
            CameraInfo, '/framos/camera_info', self.camera_info_callback, 10)        
        self.sub_goal_pose_pub = self.create_publisher(
            PoseStamped,'/sub_goal_pose', 10)
        
        self.bridge = CvBridge()
        self.image_raw_data = np.zeros(shape=(self.image_dim[0],
                                        self.image_dim[1],1),
                                        dtype=np.uint8)
        self.depth_image_raw_data = np.zeros(shape=(self.image_dim[0],
                                                      self.image_dim[1],3),
                                                      dtype=np.uint8)
        self.depth_camera_info_data = None
        self.goal_pose_data = np.zeros(shape=(7,1),dtype=np.float32)
        self.odoms_filtered = np.zeros(shape=(7,1),dtype=np.float32)
        
        # gym initializations
        # Defining action space where max subgoal position is 0.5 and 
        # max subgoal orientation is 0.785398 radians
        action_low = np.concatenate((np.ones(3)*-0.5,np.ones(4)*-0.785398))
        action_high = np.concatenate((np.ones(3)*0.5,np.ones(4)*0.785398))
        self.action_space = spaces.Box(low=action_low, high=action_high,
                                       shape=(7,) ,dtype=np.float32)

        # Defining observation space which is the output from 
        # GailNavigationNetwork  NaviNet
        # The channel shape are taken from the output of the NaviNet
        self.observation_space = spaces.Dict({
            'target_vector': spaces.Box(low=-100.0, high=100.0, shape=(1,7), dtype=np.float32),
            'rgb_features': spaces.Box(low=-np.inf, high=np.inf, shape=(1280, 8, 10), dtype=np.float32),
            'depth_image': spaces.Box(low=-np.inf, high=np.inf, shape=(2,318), dtype=np.float32)
        })


        self.model= NaviNet()

        
    def depth_image_raw_callback(self, msg):
        depth_image_raw_data = self.bridge.imgmsg_to_cv2(msg, 
                                                desired_encoding="passthrough")
        ## The scale of depth pixels is 0.001|  16bit depth, one unit is 1 mm | taken from data sheet 
        self.depth_image_raw_data = np.array(depth_image_raw_data, 
                                             dtype=np.uint16)*0.001
    
    def image_raw_callback(self, msg):
        self.image_raw_data = self.bridge.imgmsg_to_cv2(msg,
                                                        cv2.COLOR_BGR2RGB)
                
    def odometry_filtered_callback(self, msg):
        odom_data=[msg.pose.pose.position.x, 
                    msg.pose.pose.position.y, 
                    msg.pose.pose.position.z,
                    msg.pose.pose.orientation.x, 
                    msg.pose.pose.orientation.y,
                    msg.pose.pose.orientation.z,
                    msg.pose.pose.orientation.w]
        self.odoms_filtered = np.array(odom_data,dtype=np.float32)
        
    def goal_pose_callback(self, msg):
        goal_data=[msg.pose.pose.position.x, 
                    msg.pose.pose.position.y, 
                    msg.pose.pose.position.z,
                    msg.pose.pose.orientation.x, 
                    msg.pose.pose.orientation.y,
                    msg.pose.pose.orientation.z,
                    msg.pose.pose.orientation.w]
        self.goal_pose_data = np.array(goal_data,dtype=np.float32)
    def camera_info_callback(self, msg):
        self.depth_camera_info_data = msg
        self.depth_camera_height = msg.height
        self.depth_camera_width = msg.width
        self.color_camera_height = msg.height
        self.color_camera_width = msg.width

    def _get_obs(self):
        '''
        return the observation of the environment. 
        
        Return 
        ========
        the image from the camera
        '''

        rgb_features, depth_features = self.model(self.image_raw_data,
                                                  self.depth_image_raw_data)
        observation = {
            'target_vector': self.goal_pose_data,
            'rgb_features': rgb_features.numpy(),
            'depth_image': depth_features.numpy()
        }
        return observation
    def _take_action(self,action):
        return NotImplementedError

    def _get_reward(self):
        reward=0.0 
        return reward
    
    def _is_done(self,observation):
        target_vector = observation['target_vector']
        if np.linalg.norm(target_vector) < 0.01:
            return True
        else:
            return False
    
    def do_action(self,action):
        '''
        Does ROS actions based on the action passed
        '''
        done=True
        sub_goal_pose_msg = PoseStamped()
        sub_goal_pose_msg.pose.position.x = action[0].item()
        sub_goal_pose_msg.pose.position.y = action[1].item()
        sub_goal_pose_msg.pose.position.z = 0.0

        # Assign orientation
        sub_goal_pose_msg.pose.orientation.x = action[3].item()
        sub_goal_pose_msg.pose.orientation.y = action[4].item()
        sub_goal_pose_msg.pose.orientation.z = action[5].item()
        sub_goal_pose_msg.pose.orientation.w = action[6].item()

        self.sub_goal_pose_pub(sub_goal_pose_msg)
        return done

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # We need to reset the environment to its initial state
    def step(self, action: Any) :
        '''
        
        '''
        rclpy.spin_once(self)

        done=self.do_action(action)
        observation = self._get_obs()
        reward = self._get_reward()
        terminated = self._is_done(observation)
        info={}
        return observation, reward, terminated, False, info
    
    def render(self):
        return NotImplementedError
    
    def close(self):
        '''
            close any open resources that were used by the environment. eg
            eg: close the connection to the robot,close windows etc

        '''
        return NotImplementedError

# def main(args=None):
#     rclpy.init(args=args)

#     bag_reader_node = KrisEnv()

#     rclpy.spin(bag_reader_node)

#     bag_reader_node.destroy_node()
#     rclpy.shutdown()


# if __name__ == '__main__':
#     main()
