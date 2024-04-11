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
# from imitation.data.types import Trajectory
# from imitation.data import rollout
# from gymnasium.wrappers import TimeLimit
from imitation.data.wrappers import RolloutInfoWrapper
import rclpy
import numpy as np
# import h5py
# from GailNavigationNetwork.model import NaviNet
# from GailNavigationNetwork.utilities import preprocess



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
   
    # Create a vectorized environment for training with `imitation`
    env = make_vec_env(
        "kris_envs/KrisEnv-v1",
        rng=np.random.default_rng(),
        n_envs=no_envs,
        post_wrappers=[lambda env, _: RolloutInfoWrapper(env)],
    )
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
    print(f"[rl_train] venv obs shape {env.observation_space}")
    gail_trainer = GAIL(
        demonstrations=rollouts,
        demo_batch_size=24,
        gen_replay_buffer_capacity=512,
        n_disc_updates_per_round=8,
        venv=env,
        gen_algo=learner,
        reward_net=reward_net,
        allow_variable_horizon=False,
    )

    env.seed(SEED)
    # learner_rewards_before_training, _ = evaluate_policy(
    #     learner, env, 100, return_episode_rewards=True
    # )

    gail_trainer.train(2048)

if __name__ == "__main__":
    file_path="/home/foxy_user/foxy_ws/src/gail_navigation/GailNavigationNetwork/data/traj2.hdf5"
    traj_generator=TrajFromFile(file_path)
    batch_size,demonstrations=traj_generator.create_demos()
    train_gail(demonstrations)
