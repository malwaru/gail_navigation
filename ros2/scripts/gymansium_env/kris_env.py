#/usr/bin/python3 
from typing import Any, SupportsFloat
import gymnasium as gym
from gymnasium.envs.registration import register
import rclpy
import rclpy.node as Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
from GailNavigationNetwork.model import NaviNet


register(
     id="Kris-v0",
     entry_point="kris_env:KrisEnv",
     max_episode_steps=300,
)
class KrisEnv(gym.Env,Node):
    def __init__(self) -> None:
        super().__init__('KrisEnv')

        #ROS initializations
        self.image_dim=None

    def _get_obs(self):
        '''
        return the observation of the environment. 
        
        Return 
        ========
        the image from the camera
        '''
        
        return NotImplementedError
    def _take_action(self,action):
        pass

    def _get_reward(self):
        reward=0.0 
        return reward
    
    def _is_done(self):
        return NotImplementedError
    
    def do_action(self,action):
        '''
        Does ROS actions based on the action passed
        '''
        done=True
        return done

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # We need to reset the environment to its initial state
    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        '''
        
        '''
        done=self.do_action(action)
        observation = self._get_obs()
        reward = self._get_reward()
        terminated = self._is_done()
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
    