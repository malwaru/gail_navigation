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
from gail_navigation.gazebo_connection import GazeboConnection
import gymnasium as gym
import numpy as np
from gymnasium import spaces



class KrisEnv(gym.Env,Node):
    def __init__(self):
        super(KrisEnv,self).__init__('kris_env_node')

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
        
        self._cvbridge = CvBridge()
        self.gazebo = GazeboConnection()
        self.image_raw_data = None
        self.depth_image_raw_data = None
        self.depth_camera_info_data = None
        self.goal_pose_data = np.zeros(shape=(1,7),dtype=np.float32)
        self.odoms_filtered = np.zeros(shape=(1,7),dtype=np.float32)
        self.target_vector = np.zeros(shape=(1,7),dtype=np.float32)

        while self.image_raw_data is None:
            self.get_logger().info("Waiting for camera feed")
            rclpy.spin_once(self)


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
        self.observation_space = spaces.Dict({
            'target_vector': spaces.Box(low=-100.0, high=100.0, shape=(1,7), dtype=np.float32),
            'rgb_features': spaces.Box(low=-np.inf, high=np.inf, shape=(1280, 8, 10), dtype=np.float32),
            'depth_features': spaces.Box(low=-np.inf, high=np.inf, shape=(238,318), dtype=np.float32)
        })


        self.model= NaviNet()
        self.model.eval()

        
    def depth_image_raw_callback(self, msg):
        depth_image_raw_data = self._cvbridge.imgmsg_to_cv2(msg, 
                                                desired_encoding="passthrough")
        ## The scale of depth pixels is 0.001|  16bit depth, one unit is 1 mm | taken from data sheet 
        self.depth_image_raw_data = np.array(depth_image_raw_data, 
                                             dtype=np.uint16)*0.001
    
    def image_raw_callback(self, msg):
        self.image_raw_data = cv2.cvtColor(self._cvbridge.imgmsg_to_cv2(msg),
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
        rgb_image=preprocess(self.image_raw_data)
        depth_image=preprocess(self.depth_image_raw_data)  
        rgb_features, depth_features = self.model(rgb_image,
                                                  depth_image)
        # self.get_logger().info(f"depth feature shape {depth_features.shape}")
        self.target_vector = self.goal_pose_data - self.odoms_filtered
        self.get_logger().info(f"target vector shape {self.target_vector.shape}")
        observation = {
            'target_vector': self.target_vector,
            'rgb_features': rgb_features.detach().cpu().numpy(),
            'depth_features': depth_features.detach().cpu().numpy()
        }
        return observation
    
    def _take_action(self,pose):
        self.sub_goal_pose_pub(pose)
   ## COuld also be a service call depending on the robot
        done=True      
        return done

    def _get_reward(self):
        reward=0.0 
        return reward
    
    def _is_done(self,observation):
        target_vector = observation['target_vector']
        if np.linalg.norm(target_vector) < 0.01:
            return True
        else:
            return False
        
    def _get_info(self):
        target_progress = self.goal_pose_data - self.odoms_filtered
        info = {'target_vector':np.linalg.norm(target_progress)}

        return info
    
    def do_action(self,action):
        '''
        Does ROS actions based on the action passed
        '''
     
        sub_goal_pose_msg = PoseStamped()
        sub_goal_pose_msg.pose.position.x = action[0].item()
        sub_goal_pose_msg.pose.position.y = action[1].item()
        sub_goal_pose_msg.pose.position.z = 0.0

        # Assign orientation
        sub_goal_pose_msg.pose.orientation.x = action[3].item()
        sub_goal_pose_msg.pose.orientation.y = action[4].item()
        sub_goal_pose_msg.pose.orientation.z = action[5].item()
        sub_goal_pose_msg.pose.orientation.w = action[6].item()

        done=self._take_action(sub_goal_pose_msg)

        
        return done

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset()
        self.gazebo.reset_sim()

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
        raise NotImplementedError

