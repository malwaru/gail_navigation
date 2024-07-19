from stable_baselines3 import PPO
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




def test_gail(model_path,demo=None):
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
    traj=[]
    for _ in range(1000):
        action, _states = expert.predict(obs)
        obs, rewards, dones, info = vec_env.step(action)
    print("Finished reaching goal expert policy")

  

        
    

if __name__ == "__main__":
    model_path="../../GailNavigationNetwork/data/models/PPO_KrisEnv-v6_total"
    # demo_path="../../GailNavigationNetwork/data/trajectories/medium_world/traj1.hdf5"
    print(f"Model : {model_path} loaded and ready to test ") 
    test_gail(model_path)
