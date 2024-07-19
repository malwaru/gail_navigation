from gymnasium.envs.registration import register
## Registering the environments

# The dictionary observation space environment
register(
    id="kris_envs/KrisEnv-v0",
    entry_point="kris_envs.envs:KrisEnv",
    max_episode_steps=300,   
)

# The flattened obs as tuple observation space environment

register(
    id="kris_envs/KrisEnv-v1",
    entry_point="kris_envs.envs:KrisEnvTuple",
    max_episode_steps=300,   
)

# Same as above but for testing the agent 

register(
    id="kris_envs/KrisEnv-v1-2",
    entry_point="kris_envs.envs:KrisEnvTupleTest",
    max_episode_steps=300,   
)


