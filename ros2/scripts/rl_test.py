from stable_baselines3 import PPO
# import gymnasium as gym
from imitation.util.util import make_vec_env
import kris_envs
from kris_envs.wrappers.trajgen import TrajFromFile
from imitation.algorithms.adversarial.gail import GAIL
from imitation.rewards.reward_nets import BasicRewardNet,CnnRewardNet
from imitation.util.networks import RunningNorm
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy, CnnPolicy
from imitation.data.wrappers import RolloutInfoWrapper
from stable_baselines3.common.evaluation import evaluate_policy

import rclpy
import numpy as np




def test_gail(model_path):
    '''
    Trains the GAIL model
    Args:
    rollouts: gymnasium.Transition

    Returns:
    None
    '''    
    # Create custom environment
    rclpy.init()
   
    # Create a vectorized environment for training with `imitation`
    env = make_vec_env(
        "kris_envs/KrisEnv-v1-2",
        rng=np.random.default_rng(),
        n_envs=1,
        post_wrappers=[lambda env, _: RolloutInfoWrapper(env)],
    )

    # Load expert polic
    expert = PPO.load(model_path, env=env)

    vec_env = expert.get_env()
    obs = vec_env.reset()
    for _ in range(1000):
        action, _states = expert.predict(obs)
        obs, rewards, dones, info = vec_env.step(action)

    print("Finished reaching goal expert policy")
  

        
    

if __name__ == "__main__":
    model_path="../../GailNavigationNetwork/data/models/PPO_KrisEnv-v1"
    # traj_generator=TrajFromFile(file_path)
    # batch_size,demonstrations=traj_generator.create_demos()
    test_gail(model_path)
