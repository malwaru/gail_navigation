import gymnasium as gym
from stable_baselines3 import SAC, PPO
from stable_baselines3.common.evaluation import evaluate_policy
from kris_envs.envs.kris_env import KrisEnv
import rclpy

# Create custom environment
rclpy.init()
env = KrisEnv()

# Instantiate SAC model
model = SAC("MultiInputPolicy", env, verbose=1, ent_coef='auto')#,  use_sde=True, learning_rate=0.001)

# Train the model
model.learn(total_timesteps=int(1e6), progress_bar=True)

# Save the trained model
model.save("sac_custom_robot_env")

# Evaluate the trained model
mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=10, render=False)
print(f"Mean reward: {mean_reward}")
