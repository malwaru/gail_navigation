
import collections
from typing import (
    Any,    
    Dict,
    Hashable, 
    List,
    Mapping,    
    Sequence,    
    Union,
)

import numpy as np
from gymnasium import spaces
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.utils import check_for_correct_spaces
from stable_baselines3.common.vec_env import VecEnv
from imitation.data import types
from GailNavigationNetwork.model import NaviNet
from GailNavigationNetwork.utilities import preprocess


class TrajectoryAccumulator:
    """Accumulates trajectories step-by-step.

    Useful for collecting completed trajectories while ignoring partially-completed
    trajectories (e.g. when rolling out a VecEnv to collect a set number of
    transitions). Each in-progress trajectory is identified by a 'key', which enables
    several independent trajectories to be collected at once. They key can also be left
    at its default value of `None` if you only wish to collect one trajectory.
    """

    def __init__(self):
        """Initialise the trajectory accumulator."""
        self.partial_trajectories = collections.defaultdict(list)

    def add_step(
        self,
        step_dict: Mapping[str, Union[types.Observation, Mapping[str, Any]]],
        key: Hashable = None,
    ) -> None:
        """Add a single step to the partial trajectory identified by `key`.

        Generally a single step could correspond to, e.g., one environment managed
        by a VecEnv.

        Args:
            step_dict: dictionary containing information for the current step. Its
                keys could include any (or all) attributes of a `TrajectoryWithRew`
                (e.g. "obs", "acts", etc.).
            key: key to uniquely identify the trajectory to append to, if working
                with multiple partial trajectories.
        """
        self.partial_trajectories[key].append(step_dict)

    def finish_trajectory(
        self,
        key: Hashable,
        terminal: bool,
    ) -> types.TrajectoryWithRew:
        """Complete the trajectory labelled with `key`.

        Args:
            key: key uniquely identifying which in-progress trajectory to remove.
            terminal: trajectory has naturally finished (i.e. includes terminal state).

        Returns:
            traj: list of completed trajectories popped from
                `self.partial_trajectories`.
        """
        part_dicts = self.partial_trajectories[key]
        del self.partial_trajectories[key]
        out_dict_unstacked = collections.defaultdict(list)
        for part_dict in part_dicts:
            for k, array in part_dict.items():
                out_dict_unstacked[k].append(array)

        out_dict_stacked = {
            k: types.stack_maybe_dictobs(arr_list)
            for k, arr_list in out_dict_unstacked.items()
        }
        traj = types.TrajectoryWithRew(**out_dict_stacked, terminal=terminal)
        assert traj.rews.shape[0] == traj.acts.shape[0] == len(traj.obs) - 1
        return traj

    def add_steps_and_auto_finish(
        self,
        acts: np.ndarray,
        obs: Union[types.Observation, Dict[str, np.ndarray]],
        rews: np.ndarray,
        dones: np.ndarray,
        infos: List[dict],
    ) -> List[types.TrajectoryWithRew]:
        """Calls `add_step` repeatedly using acts and the returns from `venv.step`.

        Also automatically calls `finish_trajectory()` for each `done == True`.
        Before calling this method, each environment index key needs to be
        initialized with the initial observation (usually from `venv.reset()`).

        See the body of `util.rollout.generate_trajectory` for an example.

        Args:
            acts: Actions passed into `VecEnv.step()`.
            obs: Return value from `VecEnv.step(acts)`.
            rews: Return value from `VecEnv.step(acts)`.
            dones: Return value from `VecEnv.step(acts)`.
            infos: Return value from `VecEnv.step(acts)`.

        Returns:
            A list of completed trajectories. There should be one trajectory for
            each `True` in the `dones` argument.
        """
        trajs: List[types.TrajectoryWithRew] = []
        wrapped_obs = types.maybe_wrap_in_dictobs(obs)

        # iterate through environments
        for env_idx in range(len(wrapped_obs)):
            assert env_idx in self.partial_trajectories
            assert list(self.partial_trajectories[env_idx][0].keys()) == ["obs"], (
                "Need to first initialize partial trajectory using "
                "self._traj_accum.add_step({'obs': ob}, key=env_idx)"
            )

        # iterate through steps
        zip_iter = enumerate(zip(acts, wrapped_obs, rews, dones, infos))
        for env_idx, (act, ob, rew, done, info) in zip_iter:
            if done:
                # When dones[i] from VecEnv.step() is True, obs[i] is the first
                # observation following reset() of the ith VecEnv, and
                # infos[i]["terminal_observation"] is the actual final observation.
                real_ob = types.maybe_wrap_in_dictobs(info["terminal_observation"])
            else:
                real_ob = ob

            self.add_step(
                dict(
                    acts=act,
                    rews=rew,
                    # this is not the obs corresponding to `act`, but rather the obs
                    # *after* `act` (see above)
                    obs=real_ob,
                    infos=info,
                ),
                env_idx,
            )
            if done:
                # finish env_idx-th trajectory
                new_traj = self.finish_trajectory(env_idx, terminal=True)
                trajs.append(new_traj)
                # When done[i] from VecEnv.step() is True, obs[i] is the first
                # observation following reset() of the ith VecEnv.
                self.add_step(dict(obs=ob), env_idx)
        return trajs


class TrajFromFile:

    def __init__(self,file_path,DEVICE="cuda"):
        self.file_path=file_path
        self.read_file= h5py.File(file_path, "r")
        self.model= NaviNet().to(DEVICE)
        self.model.eval()
        self.len= self.read_file['kris_dynamics']['odom_data']['target_vector'].shape[0]
        self.rgbs=[]
        self.depths=[]
        self.targets=[]  
        self.acts=[]
        self.ob=[]
        self.DEVICE=DEVICE
        ## Here get the dataset propetiees so features like sample_until can be used

    def get_demo(self,idx):
        ''''
        Return the demonstration at the given index
        '''
        target=self.read_file['kris_dynamics']['odom_data']['target_vector'][idx]
        rgb=self.read_file['images']['rgb_data'][idx]
        depth=self.read_file['images']['depth_data'][i]
        act=self.read_file['kris_dynamics']['odom_data']['odom_data_wheel'][idx]
        rgb=preprocess(rgb)
        depth=preprocess(depth)
        (rgb, depth) = (rgb.to(self.DEVICE), depth.to(self.DEVICE))
        rgb_features, depth_features = self.model(rgb,depth)
        rgb_features=rgb_features.detach().cpu().numpy()
        depth_features=depth_features.detach().cpu().numpy()

        obs={
            'target_vector': np.array([target], dtype=np.float32),
            'rgb_features': np.array([rgb_features], dtype=np.float32),
            'depth_features': np.array([depth_features], dtype=np.float32)
        }

        rew=0
        done=[False if self.len-1!=idx else True]
        info={}


        



        return act,obs,rew,done,info
    
    def generate_trajectories(
        self,            
        venv: VecEnv,
        sample_until,     
    ) -> Sequence[types.TrajectoryWithRew]:
        """Generate trajectory dictionaries from a policy and an environment.

        Args:
            policy: Can be any of the following:
                1) A stable_baselines3 policy or algorithm trained on the gym environment.
                2) A Callable that takes an ndarray of observations and returns an ndarray
                of corresponding actions.
                3) None, in which case actions will be sampled randomly.
            venv: The vectorized environments to interact with.
            sample_until: An integer defining lenth of recored trajectory
            deterministic_policy: If True, asks policy to deterministically return
                action. Note the trajectories might still be non-deterministic if the
                environment has non-determinism!
            rng: used for shuffling trajectories.

        Returns:
            Sequence of trajectories, satisfying `sample_until`. Additional trajectories
            may be collected to avoid biasing process towards short episodes; the user
            should truncate if required.
        """
        # Replace this with the function to get the saved trjacotries
        # But output in the same format as the policy_to_callable
        # get_actions = policy_to_callable(policy, venv, deterministic_policy)

        # Collect rollout tuples.
        trajectories = []
        # accumulator for incomplete trajectories
        trajectories_accum = TrajectoryAccumulator()

        wrapped_obs = types.maybe_wrap_in_dictobs(obs)

        # we use dictobs to iterate over the envs in a vecenv
        for ob in (wrapped_obs):
            # Seed with first obs only. Inside loop, we'll only add second obs from
            # each (s,a,r,s') tuple, under the same "obs" key again. That way we still
            # get all observations, but they're not duplicated into "next obs" and
            # "previous obs" (this matters for, e.g., Atari, where observations are
            # really big).
            ## Here I replacec the env_idx with 1 as there is only a single idx 
            trajectories_accum.add_step(dict(obs=ob),1)

        # Now, we sample until `sample_until(trajectories)` is true.
        # If we just stopped then this would introduce a bias towards shorter episodes,
        # since longer episodes are more likely to still be active, i.e. in the process
        # of being sampled from. To avoid this, we continue sampling until all epsiodes
        # are complete.
        #
     
        dones = np.zeros(venv.num_envs, dtype=bool)
        
        for idx in range(len(sample_until)):
            acts, obs , rews , dones , infos= self.get_demo(idx)
            # obs, rews, dones, infos = venv.step(acts)    
            wrapped_obs = types.maybe_wrap_in_dictobs(obs)

            # If an environment is inactive, i.e. the episode completed for that
            # environment after `sample_until(trajectories)` was true, then we do
            # *not* want to add any subsequent trajectories from it. We avoid this
            # by just making it never done.
            # dones &= active

            new_trajs = trajectories_accum.add_steps_and_auto_finish(
                acts,
                wrapped_obs,
                rews,
                dones,
                infos,
            )
            trajectories.extend(new_trajs)





        # Note that we just drop partial trajectories. This is not ideal for some
        # algos; e.g. BC can probably benefit from partial trajectories, too.

        # Each trajectory is sampled i.i.d.; however, shorter episodes are added to
        # `trajectories` sooner. Shuffle to avoid bias in order. This is important
        # when callees end up truncating the number of trajectories or transitions.
        # It is also cheap, since we're just shuffling pointers.
        # rng.shuffle(trajectories)  # type: ignore[arg-type]

        # Sanity checks.
        for trajectory in trajectories:
            n_steps = len(trajectory.acts)
            # extra 1 for the end
            if isinstance(venv.observation_space, spaces.Dict):
                exp_obs = {}
                for k, v in venv.observation_space.items():
                    assert v.shape is not None
                    exp_obs[k] = (n_steps + 1,) + v.shape
            else:
                obs_space_shape = venv.observation_space.shape
                assert obs_space_shape is not None
                exp_obs = (n_steps + 1,) + obs_space_shape  # type: ignore[assignment]
            real_obs = trajectory.obs.shape
            assert real_obs == exp_obs, f"expected shape {exp_obs}, got {real_obs}"
            assert venv.action_space.shape is not None
            exp_act = (n_steps,) + venv.action_space.shape
            real_act = trajectory.acts.shape
            assert real_act == exp_act, f"expected shape {exp_act}, got {real_act}"
            exp_rew = (n_steps,)
            real_rew = trajectory.rews.shape
            assert real_rew == exp_rew, f"expected shape {exp_rew}, got {real_rew}"

        return trajectories

