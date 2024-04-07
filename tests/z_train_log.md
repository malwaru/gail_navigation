 # Log of taining 

 ## Trial 1 

 ### Settings 

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
    reward_net = BasicRewardNet(
        observation_space=env.observation_space,
        action_space=env.action_space,
        normalize_input_layer=RunningNorm,
    )
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

###  Log
 
 
 
 
 
 
 
 
 
 /home/foxy_user/venvs/thesis_venv/bin/python3 /home/foxy_user/foxy_ws/src/gail_navigation/ros2/scripts/rl_train.py
Available device is cuda of name NVIDIA GeForce RTX 4060 Laptop GPU

[rl_train] obs array shape (25, 178091)
/home/foxy_user/venvs/thesis_venv/lib/python3.8/site-packages/gymnasium/envs/registration.py:513: DeprecationWarning: WARN: The environment kris_envs/KrisEnv-v1 is out of date. You should consider upgrading to version `v3`.
  logger.deprecation(
[INFO] [1712146682.275416806] [kris_env_node]: Waiting for camera feed
[WARN] [1712146682.393244686] [rcl.logging_rosout]: Publisher already registered for provided node name. If this is due to multiple nodes with the same name then all logs for that logger name will go out over the existing publisher. As soon as any node with that name is destructed it will unregister the publisher, preventing any further logs for that name from being published on the rosout topic.
[WARN] [1712146682.397318868] [rcl.logging_rosout]: Publisher already registered for provided node name. If this is due to multiple nodes with the same name then all logs for that logger name will go out over the existing publisher. As soon as any node with that name is destructed it will unregister the publisher, preventing any further logs for that name from being published on the rosout topic.
[INFO] [1712146682.400814946] [kris_env_node]: Waiting for camera feed
[INFO] [1712146682.401571744] [kris_env_node]: Waiting for camera feed
[INFO] [1712146682.428773911] [kris_env_node]: Waiting for camera feed
[INFO] [1712146682.439777544] [kris_env_node]: Waiting for camera feed
[INFO] [1712146682.442005392] [kris_env_node]: Waiting for camera feed
[INFO] [1712146682.594029965] [gazebo_connection_node]: /gazebo/reset_simulation service call successful
[rl_train] Training GAIL with 24 rollouts
[rl_train] Defining rewardnet
[rl_train] Entering GAIL training loop 
[rl_train] venv obs shape Box(-inf, inf, (178091,), float32)
[INFO] [1712146683.122541386] [gazebo_connection_node]: /gazebo/reset_simulation service call successful
round:   0%|                                                                                                                                              | 0/1 [00:00<?, ?it/s][INFO] [1712146683.208792601] [gazebo_connection_node]: /gazebo/reset_simulation service call successful
[INFO] [1712146702.290630992] [gazebo_connection_node]: /gazebo/reset_simulation service call successful
[INFO] [1712146719.705624373] [gazebo_connection_node]: /gazebo/reset_simulation service call successful
[INFO] [1712146737.113686830] [gazebo_connection_node]: /gazebo/reset_simulation service call successful
[INFO] [1712146754.964667029] [gazebo_connection_node]: /gazebo/reset_simulation service call successful
[INFO] [1712146772.793944882] [gazebo_connection_node]: /gazebo/reset_simulation service call successful
[INFO] [1712146790.497448986] [gazebo_connection_node]: /gazebo/reset_simulation service call successful
------------------------------------------                                                                                                                                      
| raw/                        |          |
|    gen/rollout/ep_len_mean  | 300      |
|    gen/rollout/ep_rew_mean  | 0        |
|    gen/time/fps             | 16       |
|    gen/time/iterations      | 1        |
|    gen/time/time_elapsed    | 121      |
|    gen/time/total_timesteps | 2048     |
------------------------------------------
--------------------------------------------------                                                                                                                              
| raw/                                |          |
|    disc/disc_acc                    | 0.5      |
|    disc/disc_acc_expert             | 1        |
|    disc/disc_acc_gen                | 0        |
|    disc/disc_entropy                | 0.686    |
|    disc/disc_loss                   | 0.675    |
|    disc/disc_proportion_expert_pred | 1        |
|    disc/disc_proportion_expert_true | 0.5      |
|    disc/global_step                 | 1        |
|    disc/n_expert                    | 24       |
|    disc/n_generated                 | 24       |
--------------------------------------------------
--------------------------------------------------                                                                                                                              
| raw/                                |          |
|    disc/disc_acc                    | 1        |
|    disc/disc_acc_expert             | 1        |
|    disc/disc_acc_gen                | 1        |
|    disc/disc_entropy                | 0.0169   |
|    disc/disc_loss                   | 0.00426  |
|    disc/disc_proportion_expert_pred | 0.5      |
|    disc/disc_proportion_expert_true | 0.5      |
|    disc/global_step                 | 1        |
|    disc/n_expert                    | 24       |
|    disc/n_generated                 | 24       |
--------------------------------------------------
--------------------------------------------------                                                                                                                              
| raw/                                |          |
|    disc/disc_acc                    | 1        |
|    disc/disc_acc_expert             | 1        |
|    disc/disc_acc_gen                | 1        |
|    disc/disc_entropy                | 2.22e-05 |
|    disc/disc_loss                   | 1.67e-06 |
|    disc/disc_proportion_expert_pred | 0.5      |
|    disc/disc_proportion_expert_true | 0.5      |
|    disc/global_step                 | 1        |
|    disc/n_expert                    | 24       |
|    disc/n_generated                 | 24       |
--------------------------------------------------
--------------------------------------------------                                                                                                                              
| raw/                                |          |
|    disc/disc_acc                    | 1        |
|    disc/disc_acc_expert             | 1        |
|    disc/disc_acc_gen                | 1        |
|    disc/disc_entropy                | 3.05e-07 |
|    disc/disc_loss                   | 7.45e-09 |
|    disc/disc_proportion_expert_pred | 0.5      |
|    disc/disc_proportion_expert_true | 0.5      |
|    disc/global_step                 | 1        |
|    disc/n_expert                    | 24       |
|    disc/n_generated                 | 24       |
--------------------------------------------------
--------------------------------------------------                                                                                                                              
| raw/                                |          |
|    disc/disc_acc                    | 1        |
|    disc/disc_acc_expert             | 1        |
|    disc/disc_acc_gen                | 1        |
|    disc/disc_entropy                | 0.000195 |
|    disc/disc_loss                   | 2.53e-05 |
|    disc/disc_proportion_expert_pred | 0.5      |
|    disc/disc_proportion_expert_true | 0.5      |
|    disc/global_step                 | 1        |
|    disc/n_expert                    | 24       |
|    disc/n_generated                 | 24       |
--------------------------------------------------
--------------------------------------------------                                                                                                                              
| raw/                                |          |
|    disc/disc_acc                    | 1        |
|    disc/disc_acc_expert             | 1        |
|    disc/disc_acc_gen                | 1        |
|    disc/disc_entropy                | 0.000174 |
|    disc/disc_loss                   | 2.22e-05 |
|    disc/disc_proportion_expert_pred | 0.5      |
|    disc/disc_proportion_expert_true | 0.5      |
|    disc/global_step                 | 1        |
|    disc/n_expert                    | 24       |
|    disc/n_generated                 | 24       |
--------------------------------------------------
--------------------------------------------------                                                                                                                              
| raw/                                |          |
|    disc/disc_acc                    | 1        |
|    disc/disc_acc_expert             | 1        |
|    disc/disc_acc_gen                | 1        |
|    disc/disc_entropy                | 2e-05    |
|    disc/disc_loss                   | 1.87e-06 |
|    disc/disc_proportion_expert_pred | 0.5      |
|    disc/disc_proportion_expert_true | 0.5      |
|    disc/global_step                 | 1        |
|    disc/n_expert                    | 24       |
|    disc/n_generated                 | 24       |
--------------------------------------------------
--------------------------------------------------                                                                                                                              
| raw/                                |          |
|    disc/disc_acc                    | 1        |
|    disc/disc_acc_expert             | 1        |
|    disc/disc_acc_gen                | 1        |
|    disc/disc_entropy                | 1.07e-07 |
|    disc/disc_loss                   | 7.45e-09 |
|    disc/disc_proportion_expert_pred | 0.5      |
|    disc/disc_proportion_expert_true | 0.5      |
|    disc/global_step                 | 1        |
|    disc/n_expert                    | 24       |
|    disc/n_generated                 | 24       |
--------------------------------------------------
--------------------------------------------------                                                                                                                              
| mean/                               |          |
|    disc/disc_acc                    | 0.938    |
|    disc/disc_acc_expert             | 1        |
|    disc/disc_acc_gen                | 0.875    |
|    disc/disc_entropy                | 0.0879   |
|    disc/disc_loss                   | 0.0849   |
|    disc/disc_proportion_expert_pred | 0.562    |
|    disc/disc_proportion_expert_true | 0.5      |
|    disc/global_step                 | 1        |
|    disc/n_expert                    | 24       |
|    disc/n_generated                 | 24       |
|    gen/rollout/ep_len_mean          | 300      |
|    gen/rollout/ep_rew_mean          | 0        |
|    gen/time/fps                     | 16       |
|    gen/time/iterations              | 1        |
|    gen/time/time_elapsed            | 121      |
|    gen/time/total_timesteps         | 2.05e+03 |
|    gen/train/approx_kl              | 0        |
|    gen/train/clip_fraction          | 0        |
|    gen/train/clip_range             | 0.2      |
|    gen/train/entropy_loss           | -9.93    |
|    gen/train/explained_variance     | 0        |
|    gen/train/learning_rate          | 0.0004   |
|    gen/train/loss                   | 1.1e+24  |
|    gen/train/n_updates              | 5        |
|    gen/train/policy_gradient_loss   | -3.4e-08 |
|    gen/train/std                    | 1        |
|    gen/train/value_loss             | 2.16e+24 |
--------------------------------------------------
round: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [02:08<00:00, 128.31s/it