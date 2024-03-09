from stable_baselines3 import PPO
from kris_envs.envs.kris_env import KrisEnv
from imitation.algorithms.adversarial.gail import GAIL
from imitation.rewards.reward_nets import BasicRewardNet,CnnRewardNet
from imitation.util.networks import RunningNorm
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy, CnnPolicy
from imitation.data.types import Trajectory,DictObs
from imitation.data import rollout
from gymnasium.wrappers import TimeLimit
from imitation.data.wrappers import RolloutInfoWrapper
from stable_baselines3.common.vec_env import DummyVecEnv
import rclpy
import numpy as np
import h5py
from GailNavigationNetwork.model import NaviNet
from GailNavigationNetwork.utilities import preprocess
# import torch
# from torchvision.transforms import v2

def _make_env(max_ep_steps=500):
    """
    Helper function to create a single environment. For imitation 
     library in rolloutwrapper datatype
    """
    _env = KrisEnv()
    _env = TimeLimit(_env, max_episode_steps=max_ep_steps)
    _env = RolloutInfoWrapper(_env)
    return _env

# def preprocess(rgb_image,depth_image):

#     rgb_image =  torch.from_numpy(rgb_image)
#     rgb_image=torch.permute(rgb_image, (2, 0, 1))
#     depth_image =  np.expand_dims(depth_image, axis=0)
#     depth_image =  torch.from_numpy(depth_image)

#     rgb_transform =  v2.Compose([                      
#                         v2.ToDtype(torch.float32, scale=True),
#                         v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#                     ])
#     depth_transform = v2.Compose([                      
#                         v2.ToDtype(torch.float32, scale=True),
#                         v2.Normalize(mean=[0.485], std=[0.229]),
#                     ])
#     rgb_image = rgb_transform(rgb_image)
#     depth_image = depth_transform(depth_image)
#     return rgb_image,depth_image

def create_demos(file_path,DEVICE="cuda"):
    '''
    Creates a gymnasium transition from the given file path
    of hdf5 file of known structure

    Args:
    file_path: str  
    Path to the hdf5 file   

    Returns:
    rollouts: gymnasium.Transition

    '''    
    read_file= h5py.File(file_path, "r")
    model= NaviNet().to(DEVICE)
    model.eval()
    len= read_file['kris_dynamics']['odom_data']['target_vector'].shape[0]
    rgbs=[]
    depths=[]
    targets=[]  
    acts=[]
    for i in range(len):
        target=read_file['kris_dynamics']['odom_data']['target_vector'][i]
        rgb=read_file['images']['rgb_data'][i]
        depth=read_file['images']['depth_data'][i]
        act=read_file['kris_dynamics']['odom_data']['odom_data_wheel'][i]
        # print(f"depth shape in rollout {depth.shape}")
        rgb=preprocess(rgb)
        depth=preprocess(depth)
        # rgb,depth=preprocess(rgb,depth)
        (rgb, depth) = (rgb.to(DEVICE), depth.to(DEVICE))
        rgb_features, depth_features = model(rgb,depth)
        rgb_features=rgb_features.detach().cpu().numpy()
        depth_features=depth_features.detach().cpu().numpy()
        # print(f"depth feature in rollout {depth_features.shape}")
        rgbs.append(rgb_features)
        depths.append(depth_features)
        targets.append(target) 
        acts.append(act)
        

    acts=np.array(acts[:-1])
    dones=[False for i in range(len)]
    dones[-1]=True
    infos= [{} for i in range(len-1)]
    rgbs=np.array(rgbs)
    depths=np.array(depths)
    targets=np.array(targets)
    print(f"[rl_train] Creating rollouts {rgbs.shape} {depths.shape} , targets {targets.shape} acts {acts.shape}")
    obs_dict=DictObs( {'target_vector': targets,
            'rgb_features':rgbs,
            'depth_features': depths})
    traj = Trajectory(obs=obs_dict, acts=acts,infos=infos,terminal=dones)

    return rollout.flatten_trajectories([traj])



def train_gail(rollouts,no_envs=1):
    '''
    Trains the GAIL model
    Args:
    rollouts: gymnasium.Transition

    Returns:
    None
    '''    
    # Create custom environment
    rclpy.init()
    env = KrisEnv()
    env = TimeLimit(env, max_episode_steps=500)
    venv = DummyVecEnv([_make_env for _ in range(no_envs)])


    SEED = 42


    print(f"[rl_train] Training GAIL with {len(rollouts)} rollouts")
    
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
    print(f"[rl_train] Defining rewardnet")
    reward_net = BasicRewardNet(
        observation_space=env.observation_space,
        action_space=env.action_space,
        normalize_input_layer=RunningNorm,
    )
    print(f"[rl_train] Entering GAIL training loop ")
    gail_trainer = GAIL(
        demonstrations=rollouts,
        demo_batch_size=24,
        gen_replay_buffer_capacity=512,
        n_disc_updates_per_round=8,
        venv=venv,
        gen_algo=learner,
        reward_net=reward_net,
    )

    env.seed(SEED)
    # learner_rewards_before_training, _ = evaluate_policy(
    #     learner, env, 100, return_episode_rewards=True
    # )

    gail_trainer.train(10)

if __name__ == "__main__":
    file_path="/home/foxy_user/foxy_ws/src/gail_navigation/GailNavigationNetwork/data/traj2.hdf5"
    demonstrations=create_demos(file_path)
    train_gail(demonstrations)
