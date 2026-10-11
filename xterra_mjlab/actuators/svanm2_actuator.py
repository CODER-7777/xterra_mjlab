"""SvanM2 actuator for mjlab: delayed PD with a joint<->motor coordinate transform.

The SvanM2 leg is a **parallel mechanism**: a belt routes from the thigh
actuator past the knee to drive the calf, so the calf-motor angle depends on
both the calf joint *and* the thigh joint. This actuator models that physical
transmission in simulation: positions/velocities and torques are mapped through
the per-leg Jacobian ``J_theta`` / ``J_theta.T``, and per-motor PD gains are
derived from the nominal joint-space cfg gains::

    motor_pos      = J_theta @ q                         # joint -> motor
    tau_motor      = kp_motor * (motor_pos_des - motor_pos)
                   + kd_motor * (motor_vel_des - motor_vel)
    tau_joint      = J_theta.T @ tau_motor               # motor -> joint (virtual work)
    kp_motor[i]    = kp_joint[i] / gear_ratio[i]**2
    kd_motor[i]    = kd_joint[i] / gear_ratio[i]**2

The effective stiffness is ``K_eff = J_theta.T @ diag(kp_motor) @ J_theta``.
With equal configured gains k and the default coupling it is
``k * [[1,0,0], [0,1.25,0.5], [0,0.5,1]]``: the thigh diagonal is 1.25*k,
and thigh/calf off-diagonal terms are 0.5*k. Thus configured gains are not
independent effective joint gains. Damping has the same transformation. There is
NO ``2*pi`` factor here — this driver is all-radians (the firmware's ``2*pi`` is
a rad<->rev unit conversion for the motor driver that cancels on hardware).

The actuator group MUST cover all three joints of each leg (hip, thigh, calf)
so the 3x3 per-leg belt block lives in one Jacobian. mjlab's ``IdealPdActuator``
base handles the ``<motor>`` spec, the per-env gain tensors, and the shared
action-delay buffer; we override ``initialize`` (build the Jacobian + per-joint
effort limits) and ``compute`` (the transmission-space control law).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import mujoco
import mujoco_warp as mjwarp
import torch

from mjlab.actuator.actuator import ActuatorCmd
from mjlab.actuator.pd_actuator import IdealPdActuator, IdealPdActuatorCfg

if TYPE_CHECKING:
    from mjlab.entity import Entity

_JOINT_NAME_RE = re.compile(r"^(FL|FR|RL|RR)_(hip|thigh|calf)_joint$")
_STAGE_IDX = {"hip": 0, "thigh": 1, "calf": 2}


@dataclass(kw_only=True)
class SvanM2ActuatorCfg(IdealPdActuatorCfg):
    """Configuration for :class:`SvanM2Actuator`.

    ``effort_limit`` sizes the MJCF ``<motor>`` forcerange (set it to the max,
    24 N.m); the per-joint clamp used at runtime comes from ``effort_hip`` /
    ``effort_knee``. ``gear_ratio`` is per stage (hip, thigh, calf) and
    ``hfe_kfe_trans`` is the thigh->calf belt-coupling coefficient.
    """

    effort_hip: float = 12.0
    """Joint-side torque clamp for the hip and thigh joints (N.m)."""
    effort_knee: float = 24.0
    """Joint-side torque clamp for the calf joint (N.m)."""
    gear_ratio: tuple[float, float, float] = (8.0, 8.0, 16.0)
    """Per-stage motor gear ratio (hip, thigh, calf)."""
    hfe_kfe_trans: float = 0.5
    """Belt coupling: ``motor_calf = gear_calf * (q_calf + hfe_kfe_trans * q_thigh)``."""

    def build(self, entity: "Entity", target_ids: list[int], target_names: list[str]) -> "SvanM2Actuator":
        return SvanM2Actuator(self, entity, target_ids, target_names)


class SvanM2Actuator(IdealPdActuator[SvanM2ActuatorCfg]):
    """Delayed-PD actuator with the SvanM2 parallel-mechanism joint<->motor map."""

    def initialize(
        self,
        mj_model: mujoco.MjModel,
        model: mjwarp.Model,
        data: mjwarp.Data,
        device: str,
    ) -> None:
        super().initialize(mj_model, model, data, device)

        names = [n.split("/")[-1] for n in self._target_names]  # strip attach prefix
        n = len(names)

        # leg -> {stage: index-within-this-actuator}
        per_leg: dict[str, dict[str, int]] = {}
        for i, name in enumerate(names):
            m = _JOINT_NAME_RE.match(name)
            if m is None:
                raise ValueError(
                    f"SvanM2Actuator: joint {name!r} does not match "
                    "(FL|FR|RL|RR)_(hip|thigh|calf)_joint; all joints in this "
                    "actuator group must be SvanM2 leg joints."
                )
            per_leg.setdefault(m.group(1), {})[m.group(2)] = i
        for leg, stages in per_leg.items():
            missing = {"hip", "thigh", "calf"} - stages.keys()
            if missing:
                raise ValueError(f"SvanM2Actuator: leg {leg!r} missing stages {sorted(missing)}.")

        gear = self.cfg.gear_ratio
        hk = self.cfg.hfe_kfe_trans
        J = torch.zeros(n, n, dtype=torch.float, device=device)
        motor_gear = torch.zeros(n, dtype=torch.float, device=device)
        force_limit = torch.empty(n, dtype=torch.float, device=device)
        for stages in per_leg.values():
            h, t, c = stages["hip"], stages["thigh"], stages["calf"]
            J[h, h] = gear[_STAGE_IDX["hip"]]
            J[t, t] = gear[_STAGE_IDX["thigh"]]
            J[c, c] = gear[_STAGE_IDX["calf"]]
            # Belt coupling: calf motor angle depends on thigh joint angle.
            J[c, t] = gear[_STAGE_IDX["calf"]] * hk
            motor_gear[h] = gear[_STAGE_IDX["hip"]]
            motor_gear[t] = gear[_STAGE_IDX["thigh"]]
            motor_gear[c] = gear[_STAGE_IDX["calf"]]
            force_limit[h] = self.cfg.effort_hip
            force_limit[t] = self.cfg.effort_hip
            force_limit[c] = self.cfg.effort_knee

        self._J = J                      # tau_joint = tau_motor @ J
        self._JT = J.t().contiguous()    # motor_x = joint_x @ J.T
        self._motor_gear_sq = (motor_gear**2).unsqueeze(0)  # (1, n)
        # Per-joint joint-side torque clamp (overrides the scalar effort_limit).
        assert self.force_limit is not None
        self.force_limit[:] = force_limit.unsqueeze(0)
        self.default_force_limit = self.force_limit.clone()

    def compute(self, cmd: ActuatorCmd) -> torch.Tensor:
        # Joint -> motor space (positions / velocities).
        motor_pos_des = cmd.position_target @ self._JT
        motor_pos = cmd.pos @ self._JT
        motor_vel_des = cmd.velocity_target @ self._JT
        motor_vel = cmd.vel @ self._JT
        # Per-motor PD gains derived from the joint-space cfg gains (respects
        # any per-env gain randomisation applied to self.stiffness/self.damping).
        motor_kp = self.stiffness / self._motor_gear_sq
        motor_kd = self.damping / self._motor_gear_sq
        tau_motor = motor_kp * (motor_pos_des - motor_pos) + motor_kd * (motor_vel_des - motor_vel)
        # Motor -> joint torque (virtual work) + joint-space feedforward.
        tau_joint = tau_motor @ self._J + cmd.effort_target
        return torch.clamp(tau_joint, -self.force_limit, self.force_limit)
