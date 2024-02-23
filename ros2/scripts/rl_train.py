import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from kris_env.kris_env import KrisEnv
import rclpy
import numpy as np
from imitation.algorithms.adversarial.gail import GAIL
from imitation.rewards.reward_nets import BasicRewardNet
from imitation.util.networks import RunningNorm
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.evaluation import evaluate_policy
from imitation.data.types import Trajectory
import h5py
from GailNavigationNetwork.model import NaviNet

# Create custom environment
rclpy.init()
env = KrisEnv()
SEED = 42
file_path="/home/foxy_user/foxy_ws/src/gail_navigation/GailNavigationNetwork/data/traj2.hdf5"
read_file= h5py.File(file_path, "r")
model= NaviNet()
len= read_file['kris_dynamics']['odom_data']['target_vector'].shape[0]
obs=[]
acts=[]
for i in range(len):
    target=read_file['kris_dynamics']['odom_data']['target_vector'][i]
    rgb=model.forwardread_file['images']['rgb_data'][i]
    depth=read_file['images']['depth_data'][i]
    rgb_features, depth_features = model.forward(rgb,depth)
    act=read_file['kris_dynamics']['odom_data']['odom_data_wheel'][i]
    obs.append([target,rgb_features,depth_features])
 
    acts.append(act)
# print(obs)

del acts[-1]
dones=np.zeros(shape=(len,1))
dones[-1]=1
infos= [{} for i in range(len-1)]
rollouts = Trajectory(obs=obs, acts=acts,infos=infos,terminal=dones)


learner = PPO(
    env=env,
    policy=MlpPolicy,
    batch_size=64,
    ent_coef=0.0,
    learning_rate=0.0004,
    gamma=0.95,
    n_epochs=5,
    seed=SEED,
)
reward_net = BasicRewardNet(
    observation_space=env.observation_space,
    action_space=env.action_space,
    normalize_input_layer=RunningNorm,
)
gail_trainer = GAIL(
    demonstrations=rollouts,
    demo_batch_size=1024,
    gen_replay_buffer_capacity=512,
    n_disc_updates_per_round=8,
    venv=env,
    gen_algo=learner,
    reward_net=reward_net,
)

env.seed(SEED)
learner_rewards_before_training, _ = evaluate_policy(
    learner, env, 100, return_episode_rewards=True
)

gail_trainer.train(200_000)