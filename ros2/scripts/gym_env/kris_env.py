#/usr/bin/env python3
# -*- coding: utf-8 -*-

import gym
from gym import utils, spaces
from gym.utils import seeding
from gym.envs.registration import register
from gazebo_connection import GazeboConnection

import numpy as np
import rclpy
from geometry_msgs.msg import PointStamped, Twist,PoseStamped,TransformStamped
import cv2

#register the training environment in the gym as an available one
reg = register(
    id='Kris-v0',
    entry_point='kris_env:KrisEnv', # the entry point is name of pythonfile followed by class name 
    timestep_limit=100000,
    )


class KrisEnv(gym.Env):
    def __init__(self):
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:

        self.gazebo= GazeboConnection()
        
        self.action_space = spaces.Discrete(3)
        # Example for using image as input:
        self.observation_space = spaces.Box(low=0, high=255, shape=(100, 100, 3), dtype=np.uint8)
        self._seed()
        self._reset()
        self._configure()

if __name__ == '__main__':
    pass