from gymnasium.envs.registration import register
register(
    id="KrisEnv-v0",
    entry_point="kris_envs.envs:KrisEnv",
    max_episode_steps=200,
    reward_threshold=195.0,
)