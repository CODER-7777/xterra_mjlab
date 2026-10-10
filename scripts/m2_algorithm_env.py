"""Native public SvanM2 terrain task for the new algorithm comparison.

This is a fresh task, not a continuation or alteration of the old ISL gates.
Hardware, actions and actuator files are inherited unchanged from public M2.
"""
import math
from copy import deepcopy

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.envs import mdp
from mjlab.managers import TerminationTermCfg
from mjlab.terrains import config as terrains
from mjlab.terrains.terrain_generator import TerrainGeneratorCfg
from mjlab.tasks.velocity.mdp.curriculums import terrain_levels_vel
from xterra_mjlab.tasks.velocity.config.svanm2.env_cfgs import svanm2_rough_env_cfg


def terrain_curriculum(env, env_ids, command_name):
    # Construction/reset is not a completed traversal.
    if env.common_step_counter == 0 or getattr(env, "_m2_restoring", False):
        return {"mean": env.scene.terrain.terrain_levels.float().mean()}
    return terrain_levels_vel(env, env_ids, command_name)


def make_cfg(num_envs, seed):
    cfg = deepcopy(svanm2_rough_env_cfg())
    cfg.scene.num_envs = num_envs
    cfg.seed = seed
    # Explicit manual reset preserves true successor states for the HIM target.
    cfg.auto_reset = False
    cfg.observations["actor"].history_length = 1
    cfg.scene.terrain.max_init_terrain_level = 1
    cfg.scene.terrain.terrain_generator = TerrainGeneratorCfg(
        seed=seed, size=(8., 8.), num_rows=12, curriculum=True,
        border_width=10., sub_terrains={
            "flat": terrains.flat(proportion=.20),
            "stairs_up": terrains.pyramid_stairs(proportion=.25,
                step_height_range=(.02, .27), step_width=.30, platform_width=2.),
            "stairs_down": terrains.pyramid_stairs_inv(proportion=.25,
                step_height_range=(.02, .27), step_width=.30, platform_width=2.),
            "slope_up": terrains.hf_pyramid_slope(proportion=.15,
                slope_range=(0., math.tan(math.radians(32.)))),
            "slope_down": terrains.hf_pyramid_slope_inv(proportion=.15,
                slope_range=(0., math.tan(math.radians(32.)))),
        })
    cmd = cfg.commands["twist"]
    cmd.ranges.lin_vel_x = (-.5, 1.)
    cmd.ranges.lin_vel_y = (-.3, .3)
    cmd.ranges.ang_vel_z = (-.5, .5)
    cfg.curriculum["command_vel"].params["velocity_stages"] = [
        {"step": 0, "lin_vel_x": (-.5, 1.)},
        {"step": 100000, "lin_vel_x": (-.5, 1.5)},
        {"step": 300000, "lin_vel_x": (-1., 2.)},
    ]
    cfg.curriculum["terrain_levels"].func = terrain_curriculum
    # Curriculum is per-environment distance based, no all-cases level-3 lock.
    # Keep native illegal-contact termination; add a physical tilt backstop.
    cfg.terminations["fell_over"] = TerminationTermCfg(
        func=mdp.bad_orientation, params={"limit_angle": math.radians(60.)})
    # Initial learning should not be dominated by pushes every 1-3 seconds.
    cfg.events["push_robot"].interval_range_s = (10., 15.)
    cfg.events["push_robot"].params["velocity_range"] = {
        "x": (-.5, .5), "y": (-.5, .5)}
    return cfg


class Adapter:
    def __init__(self, cfg, device, history=6, frame_size=45, numerical_guard=False):
        self.numerical_guard = numerical_guard
        self.numerical_resets = 0
        self.env = ManagerBasedRlEnv(cfg, device=device)
        obs, _ = self.env.reset()
        self.length = history
        assert obs["actor"].shape == (cfg.scene.num_envs, frame_size)
        self.history = obs["actor"].unsqueeze(1).repeat(1, history, 1)
        self.critic = obs["critic"]
        assert self.critic.shape[1] >= 48

    @property
    def actor(self):
        return self.history.flatten(1)

    def step(self, actions):
        if self.numerical_guard and not torch.isfinite(actions).all():
            raise FloatingPointError("Non-finite action before simulator step")
        obs, rewards, terminated, truncated, extras = self.env.step(actions.clamp(-6., 6.))
        reward_term_rates = (
            self.env.reward_manager._step_reward.detach().mean(dim=0)
        )
        invalid = torch.zeros_like(terminated)
        if self.numerical_guard:
            invalid = (~torch.isfinite(obs["actor"]).all(1)
                       | ~torch.isfinite(obs["critic"]).all(1)
                       | ~torch.isfinite(rewards)
                       | ~torch.isfinite(self.env.sim.data.qpos).all(1)
                       | ~torch.isfinite(self.env.sim.data.qvel).all(1))
            count = int(invalid.sum())
            if count:
                if count > max(1, len(invalid) // 100):
                    raise FloatingPointError(f"Widespread simulator failure: {count}/{len(invalid)} worlds")
                self.numerical_resets += count
                print(f"NUMERICAL_RESET step={self.env.common_step_counter} envs={invalid.nonzero().flatten().tolist()}", flush=True)
                terminated = terminated | invalid
                truncated = truncated & ~invalid
                rewards = rewards.clone()
                rewards[invalid] = -1.
        done = terminated | truncated
        # Capture BEFORE manual reset: no cross-episode estimator target leakage.
        successor = obs["critic"].clone()
        self.successor_actor = obs["actor"].clone()
        if self.numerical_guard:
            # Invalid worlds are terminal failures, never timeout-bootstrapped.
            successor[invalid] = 0.
            self.successor_actor[invalid] = 0.
        rewards = rewards.clone()
        done = done.clone()
        truncated = truncated.clone()
        info = dict(extras)
        info["reward_term_rates"] = reward_term_rates
        info["terminal_events"] = terminated.float().sum()
        info["finished_episodes"] = done.float().sum()
        info["finished_episode_seconds"] = (self.env.episode_length_buf * done).sum()*self.env.step_dt
        ids = done.nonzero(as_tuple=False).flatten()
        if ids.numel():
            obs, _ = self.env.reset(env_ids=ids)
        if self.numerical_guard:
            if not torch.isfinite(obs["actor"]).all() or not torch.isfinite(obs["critic"]).all():
                raise FloatingPointError("Simulator reset did not restore finite observations")
        self.history = torch.cat((obs["actor"].unsqueeze(1), self.history[:, :-1]), 1)
        self.history[ids] = obs["actor"][ids, None].expand(-1, self.length, -1)
        self.critic = obs["critic"]
        info["time_outs"] = truncated
        return rewards, done, info, successor
