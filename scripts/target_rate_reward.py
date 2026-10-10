"""Time-aware target-rate reward, enabled by train_other.py's explicit CLI flag.

Importing this file does not modify a configuration or start training.
"""
from __future__ import annotations

import math

import torch
from mjlab.managers import RewardTermCfg

from jitter_rewards import _TargetCost


def target_rate_excess(target, previous, step_dt, speed_limits):
    """Dimensionless excess of target increment / time, summed over joints.

    Inputs are position targets in radians, seconds, and positive rad/s limits.
    Caller validates timing/limits once when constructing the manager term.
    """
    excess = torch.relu((target - previous).abs() / (step_dt * speed_limits) - 1.0)
    return excess.square().sum(dim=-1)


class TargetRateExcess(_TargetCost):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        limits = torch.as_tensor(cfg.params["speed_limits_rad_s"],
            device=env.action_manager.action.device, dtype=env.action_manager.action.dtype)
        if limits.ndim != 1 or limits.numel() != self.term.action_dim:
            raise ValueError("Supply one rad/s threshold for every action joint in target_names order")
        if not bool(torch.isfinite(limits).all()) or bool((limits <= 0).any()):
            raise ValueError("Target-rate thresholds must be positive and finite")
        if not math.isfinite(float(env.step_dt)) or env.step_dt <= 0:
            raise ValueError("Use positive policy/environment step_dt, not a wall-clock or physics dt")
        self.speed_limits = limits.unsqueeze(0)

    def __call__(self, env, speed_limits_rad_s, action_name="joint_pos"):
        del speed_limits_rad_s  # Cached at construction, like _TargetCost's scales/bounds.
        if action_name != self.action_name:
            raise ValueError("Action term changed")
        manager = env.action_manager
        cost = target_rate_excess(self.commanded(manager.action),
            self.commanded(manager.prev_action), env.step_dt, self.speed_limits)
        # The first action's predecessor is reset history, not an executed target.
        return torch.where(env.episode_length_buf >= 2, cost, torch.zeros_like(cost))


def configure_target_rate_trial(cfg, speed_limits_rad_s, *, weight=-0.05,
                                replace_second_difference=False):
    """Explicit registration; by default retain existing smoothness/saturation.

    The weight is exploratory and not comparable numerically to another cost's
    weight. Call only after config_jitter_rewards(cfg). Never register pressure
    twice; the existing target_saturation_l2 remains unchanged.
    """
    if not math.isfinite(weight) or weight >= 0:
        raise ValueError("Choose a finite negative trial weight")
    if "target_saturation_l2" not in cfg.rewards:
        raise ValueError("Call existing config_jitter_rewards first to retain saturation pressure")
    limits = [float(v) for v in speed_limits_rad_s]
    if not limits or any(not math.isfinite(v) or v <= 0 for v in limits):
        raise ValueError("Use positive finite rad/s thresholds")
    cfg.rewards["target_rate_excess_l2"] = RewardTermCfg(func=TargetRateExcess,
        weight=weight, params={"action_name": "joint_pos", "speed_limits_rad_s": limits})
    if replace_second_difference:
        cfg.rewards.pop("target_second_difference_l2", None)
    return cfg
