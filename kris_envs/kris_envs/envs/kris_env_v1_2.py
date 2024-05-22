import gymnasium as gym
from gymnasium import spaces
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image,CameraInfo
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist,PoseStamped
from cv_bridge import CvBridge
import cv2
import numpy as np
from GailNavigationNetwork.model import NaviNet
from GailNavigationNetwork.utilities import preprocess
# from stable_baselines3.common.env_checker import check_env
import gymnasium as gym
import numpy as np
from gymnasium import spaces
import time
from kris_envs.wrappers.utilities import denormalise_action,transform_to_int8,\
                                        img_resize
from kris_envs.wrappers.gazebo_connection import GazeboConnection



class KrisEnvTupleTest(gym.Env,Node):
    def __init__(self):
        super(KrisEnvTupleTest,self).__init__('kris_env_node')
        #ROS initializations
        self.image_dim=(240,320)          
        self.depth_image_raw_sub = self.create_subscription(
            Image, '/framos/depth/image_raw', self.depth_image_raw_callback, 10)
        self.image_raw_sub = self.create_subscription(
            Image, '/framos/image_raw', self.image_raw_callback, 10)
        self.odometry_filtered_sub = self.create_subscription(
            Odometry, '/odometry/wheel', self.odometry_filtered_callback, 10)
        self.goal_pose_sub = self.create_subscription(
            PoseStamped, '/target_goal', self.goal_pose_callback, 10)
        self.camera_info_sub = self.create_subscription(
            CameraInfo, '/framos/camera_info', self.camera_info_callback, 10)        
        self.sub_goal_pose_pub = self.create_publisher(
            PoseStamped,'/subgoal_pose', 10)
        
        self._cvbridge = CvBridge()
        self.gazebo = GazeboConnection()
        self.image_raw_data = None
        self.depth_image_raw_data = None
        self.depth_camera_info_data = None
        self.goal_pose_data = np.zeros(shape=(1,7),dtype=np.float32)
        self.odoms_filtered = np.zeros(shape=(1,7),dtype=np.float32)
        self.target_vector = np.zeros(shape=(1,7),dtype=np.float32)
        self.target_vector_tolerance = 1.0 # meters
        self.observation_delay=1.0 # seconds to wait for the observation to be ready

        while self.image_raw_data is None:
            self.get_logger().info("Waiting for camera feed")
            rclpy.spin_once(self)

        self.get_logger().info("Camera feed received")

        while self.goal_pose_sub  is None:
            self.get_logger().info("Waiting for goal pose")
            rclpy.spin_once(self)

        self.get_logger().info("Goal pose received")
        # gym initializations
        # Defining action space where max subgoal position is 0.5 and 
        # max subgoal orientation is 0.785398 radians
        # But the action space used is normalized to [-1,1]
        action_low = np.ones(7)*-1.0
        action_high = np.ones(7)*1.0
        self.action_space = spaces.Box(low=action_low, high=action_high,
                                       shape=(7,) ,dtype=np.float32)
        

        # Defining observation space which is the output from 
        # GailNavigationNetwork  NaviNet
        # The channel shape are taken from the output of the NaviNet
        # after passing the image through the network
        # states in the order target vector, rgb_features, depth_features are fl
        # flattened and concatenated to form the observation space
        # Issues https://stable-baselines3.readthedocs.io/en/master/guide/algos.html
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(178091,), dtype=np.float32)
        self.model= NaviNet()
        self.model.eval()

        
    def depth_image_raw_callback(self, msg):
        depth_image_raw_data = self._cvbridge.imgmsg_to_cv2(msg, 
                                                desired_encoding="passthrough")
        depth_image_raw_data=img_resize(depth_image_raw_data,scale=1.0)
        depth_image_raw_data=transform_to_int8(depth_image_raw_data)
        ## The scale of depth pixels is 0.001|  16bit depth, one unit is 1 mm | taken from data sheet 
        self.depth_image_raw_data = np.array(depth_image_raw_data)
    
    def image_raw_callback(self, msg):
        image_raw_data = cv2.cvtColor(self._cvbridge.imgmsg_to_cv2(msg),
                                           cv2.COLOR_BGR2RGB)
        image_raw_data=img_resize(image_raw_data,scale=1.0)         
        self.image_raw_data = np.array(image_raw_data)
        # self.get_logger().info(f"[KrisEnv::image_raw_callback]self.image_raw_data.shape{self.image_raw_data.shape}")    
           
                
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
        rgb_image=preprocess(self.image_raw_data)
        depth_image=preprocess(self.depth_image_raw_data)  
        rgb_features, depth_features = self.model(rgb_image,
                                                  depth_image)
        self.target_vector = (self.goal_pose_data - self.odoms_filtered).flatten()
        rgb_features=rgb_features.detach().cpu().numpy().flatten()
        depth_features=depth_features.detach().cpu().numpy().flatten()

        flatten_obs=np.concatenate((self.target_vector,rgb_features,depth_features))
        
        return flatten_obs
    
    def _take_action(self,pose):
        self.sub_goal_pose_pub.publish(pose)
        ## Wait for the actions to be executed completely   
        # Could also be a service call depending on the robot
        time.sleep(self.observation_delay)
        done=True      
        
        return done

    def _get_reward(self):
        reward=0.0 
        return reward
    
    def _is_done(self,observation):
        # target_vector = observation['target_vector']
        if np.linalg.norm(self.target_vector) < self.target_vector_tolerance:
            return True
        else:
            return False
        
    def _get_info(self):
        target_progress = self.goal_pose_data - self.odoms_filtered
        info = {'goal pose' : self.goal_pose_data,
                'current_pose':self.odoms_filtered,
                'target_progress':target_progress,}
        return info
    
    def do_action(self,action):
        '''
        Does ROS actions based on the action passed
        '''
        denorm_action=np.array([action[0].item(),action[1].item(),action[2].item(),
                                action[3].item(),action[4].item(),action[5].item(),
                                action[6].item()])
        denorm_action=denormalise_action(denorm_action)
     
        sub_goal_pose_msg = PoseStamped()
        sub_goal_pose_msg.header.frame_id = 'base_link'
        sub_goal_pose_msg.pose.position.x = denorm_action[0]
        sub_goal_pose_msg.pose.position.y = denorm_action[1]
        sub_goal_pose_msg.pose.position.z = denorm_action[2]

        # Assign orientation
        sub_goal_pose_msg.pose.orientation.x = denorm_action[3]
        sub_goal_pose_msg.pose.orientation.y = denorm_action[4]
        sub_goal_pose_msg.pose.orientation.z = denorm_action[5]
        sub_goal_pose_msg.pose.orientation.w = denorm_action[6]

        done=self._take_action(sub_goal_pose_msg)

        
        return done

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset()
        new_obs = self._get_obs()

        return new_obs,{}

        # We need to reset the environment to its initial state
    def step(self, action) :
        '''
        Take a step in the environment
        Args: It takes an action and returns the observation, reward,

        Returns: observation, reward, terminated, info
        '''
        rclpy.spin_once(self)
        done=self.do_action(action)
        observation = self._get_obs()
        reward = self._get_reward()
        terminated = self._is_done(observation)
        info=self._get_info()
        return observation, reward, terminated, False, info
    
    def render(self):
        raise NotImplementedError
    
    def close(self):
        '''
            close any open resources that were used by the environment. eg
            eg: close the connection to the robot,close windows etc

        '''
        print("Closing the environment")
