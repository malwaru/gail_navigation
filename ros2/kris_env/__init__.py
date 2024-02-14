from gymnasium.envs.registration import register

register(
     id="Kris-v0",
     entry_point="kris_env:KrisEnv",
     max_episode_steps=300,
)