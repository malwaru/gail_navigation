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
from typing import Callable


import rclpy
import numpy as np


def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        return progress_remaining * initial_value

    return func

def train_gail(rollouts,demo_batch_size,no_envs=1):
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
        batch_size=64, # Mini batch size
        ent_coef=0.0,  # Entropy coefficient for the loss calculation
        learning_rate=linear_schedule(0.0009),
        gamma=0.95, # Discount factor
        n_epochs=5, # Number of epochs when optimizing the surrogate objective
        seed=SEED,
    )
    print(f"[rl_train] Defining rewardnet")
    reward_net = BasicRewardNet(
        observation_space=env.observation_space,
        action_space=env.action_space,
        normalize_input_layer=RunningNorm,
    )
    gail_trainer = GAIL(
        demonstrations=rollouts,
        demo_batch_size=demo_batch_size-1, # Batch size of expert demonstrations
        gen_replay_buffer_capacity=512, # Capacity of the replay buffer number of obs-action-obs samples from the generator that can be stored)
        n_disc_updates_per_round=8, # Number of discriminator updates per round of training
        venv=env,
        gen_algo=learner,
        reward_net=reward_net,
        allow_variable_horizon=False,
        
    )

    env.seed(SEED)
    # learner_rewards_before_training, _ = evaluate_policy(
    #     learner, env, 100, return_episode_rewards=True
    # )
    print(f"[rl_train] Entering GAIL training  ")
    gail_trainer.train(2048)
    print(f"[rl_train] Training complete")
    # learner_rewards_after_training, _ = evaluate_policy(
    # learner, env, 100, return_episode_rewards=True,)
    learner.save("../../GailNavigationNetwork/data/models/PPO_KrisEnv-v1")
    # print("mean reward after training:", np.mean(learner_rewards_after_training))
    # print("mean reward before training:", np.mean(learner_rewards_before_training))

if __name__ == "__main__":
    file_path="../../GailNavigationNetwork/data/trajectories/medium_world/traj1.hdf5"
    traj_generator=TrajFromFile(file_path)
    batch_size,demonstrations=traj_generator.create_demos_from_file()
    train_gail(demonstrations,batch_size)
