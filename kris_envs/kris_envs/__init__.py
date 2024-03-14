from gymnasium.envs.registration import register

register(
    id="kris_envs/KrisEnv-v0",
    entry_point="kris_envs.envs:KrisEnv",
    max_episode_steps=300,   
)

register(
    id="kris_envs/KrisEnv-v1",
    entry_point="kris_envs.envs:KrisEnvRgb",
    max_episode_steps=300,   
)