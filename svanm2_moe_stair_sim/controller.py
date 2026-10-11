"""MoE 90k policy controller and actuator PD driver for SvanM2 in MuJoCo.

Simulator policy/control contract (hardware limits/gains require separate validation):
- Checkpoint: model_90000.pt (loaded directly)
- 5-frame x 45-dim history buffer (oldest-first into student network)
- Action scaling [0.15, 0.15, 0.30] with joint limit clamping
- Coupled belt-drive PD torque model (kp=20.0, kd=0.7, gear=[8,8,16])
- Policy at 50 Hz, physics at 200 Hz (decimation 4)
"""

from pathlib import Path
import numpy as np
import torch
import mujoco
from checkpoint_policy import DEFAULT_CHECKPOINT, load_checkpoint_policy

JOINTS = [f"{leg}_{stage}_joint" for leg in ("FL", "FR", "RL", "RR") for stage in ("hip", "thigh", "calf")]
Q0 = np.array([0., 0.5806, -1.1716, 0., 0.5806, -1.1716, 0., 0.7167, -1.1342, 0., 0.7167, -1.1342], dtype=np.float32)
STAND_Q = Q0.copy()
SCALE = np.tile([0.15, 0.15, 0.30], 4).astype(np.float32)
TORQUE_LIMITS = np.tile([12.0, 12.0, 24.0], 4).astype(np.float32)
GEAR = np.tile([8.0, 8.0, 16.0], 4).astype(np.float32)

# Software position-target bounds; full per-joint deployment parity remains under review.
TARGET_MIN = np.array([
    -0.7350, -1.5708, -2.7227,  # FL
    -0.7350, -1.5708, -2.7227,  # FR
    -0.7350, -0.5236, -2.7227,  # RL
    -0.7350, -0.5236, -2.7227   # RR
], dtype=np.float32)

# Important target-bound corrections from the recorded deployed-adapter audit:
# calf upper -0.88776 rad, hip ±0.735 rad. These are commanded-position clamps,
# not verified physical-stop angles or a hardware firmware change.
TARGET_MAX = np.array([
     0.7350,  3.4907, -0.88776,  # FL
     0.7350,  3.4907, -0.88776,  # FR
     0.7350,  4.5379, -0.88776,  # RL
     0.7350,  4.5379, -0.88776   # RR
], dtype=np.float32)

# Belt coupling Jacobian (4 legs, 3x3 block diagonal)
J_LEG = np.array([[8.0, 0.0, 0.0],
                  [0.0, 8.0, 0.0],
                  [0.0, 8.0, 16.0]], dtype=np.float32)
J_COUPLING = np.kron(np.eye(4, dtype=np.float32), J_LEG)


class MoeStairController:
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData,
                 checkpoint_path: Path = DEFAULT_CHECKPOINT, device: str = "cpu",
                 observation_timing: str = "legacy"):
        self.model = model
        self.data = data
        if observation_timing not in ("legacy", "fresh"):
            raise ValueError("observation_timing must be legacy or fresh")
        self.observation_timing = observation_timing
        self.observation_data = mujoco.MjData(model) if observation_timing == "fresh" else None

        self.checkpoint_path = Path(checkpoint_path).expanduser().resolve()
        self.policy, self.checkpoint_update = load_checkpoint_policy(self.checkpoint_path, device)
        self.device = next(self.policy.parameters()).device
        print(f"[INFO] Loaded checkpoint: {self.checkpoint_path}")
        print(f"[INFO] Checkpoint update: {self.checkpoint_update} (student + actor)")
        print(f"[INFO] Policy device: {self.device}")
        if self.device.type == "cuda":
            print(f"[INFO] GPU: {torch.cuda.get_device_name(self.device)}")
        print("[INFO] Physics: standard MuJoCo on CPU (200 Hz); policy inference: 50 Hz")
        print(f"[INFO] Observation timing: {self.observation_timing}")

        # Cache joint indices
        self.qpos_idx = np.array([model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j)] for j in JOINTS])
        self.qvel_idx = np.array([model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j)] for j in JOINTS])
        self.base_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")
        self.gyro_sensor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "imu_ang_vel")
        self.gyro_adr = model.sensor_adr[self.gyro_sensor_id] if self.gyro_sensor_id >= 0 else -1

        # Control state
        self.kp = 18.0
        self.kd = 0.7
        self.previous_action = np.zeros(12, dtype=np.float32)
        self.target_q = STAND_Q.copy()
        
        # 5 frames of 45-dim observations (oldest-first)
        # history[0] is oldest, history[4] is newest
        self.history = np.zeros((5, 45), dtype=np.float32)
        self.initialized = False
        self.step_count = 0

    def reset(self, target_pos: tuple[float, float, float] | None = None, yaw: float = 0.0):
        """Reset robot pose and history buffer."""
        if target_pos is not None:
            # Set base position
            self.data.qpos[0:3] = target_pos
            # Set base orientation quaternion [w, x, y, z] in MuJoCo
            half_yaw = yaw * 0.5
            self.data.qpos[3:7] = [np.cos(half_yaw), 0.0, 0.0, np.sin(half_yaw)]
            self.data.qvel[:6] = 0.0

        # Set nominal joints
        self.data.qpos[self.qpos_idx] = STAND_Q
        self.data.qvel[self.qvel_idx] = 0.0
        self.target_q = STAND_Q.copy()
        self.previous_action = np.zeros(12, dtype=np.float32)
        self.history.fill(0.0)
        self.initialized = False
        self.step_count = 0
        mujoco.mj_forward(self.model, self.data)

    def get_current_frame(self, command: tuple[float, float, float]) -> np.ndarray:
        """Select legacy readings or refresh derived sensors on copied state."""
        if self.observation_timing == "fresh":
            mujoco.mj_copyData(self.observation_data, self.model, self.data)
            mujoco.mj_forward(self.model, self.observation_data)
            return self._frame_from_data(self.observation_data, command)
        return self.get_legacy_frame(command)

    def get_legacy_frame(self, command: tuple[float, float, float]) -> np.ndarray:
        """Original mixed-time readings, retained for comparisons and logging."""
        return self._frame_from_data(self.data, command)

    def _frame_from_data(self, data, command) -> np.ndarray:
        q = data.qpos[self.qpos_idx].astype(np.float32)
        dq = data.qvel[self.qvel_idx].astype(np.float32)

        # Base orientation & projected gravity
        # In MuJoCo xmat is 9 floats (row-major 3x3 rotation matrix from body to world)
        rot = data.xmat[self.base_body_id].reshape(3, 3).astype(np.float32)
        # World gravity is [0, 0, -1]. Projected gravity in body frame: rot.T @ [0, 0, -1]
        gravity_proj = rot.T @ np.array([0.0, 0.0, -1.0], dtype=np.float32)

        # Angular velocity from IMU sensor or estimated from body velocity
        if self.gyro_adr >= 0:
            gyro = data.sensordata[self.gyro_adr:self.gyro_adr + 3].astype(np.float32)
        else:
            gyro = (rot.T @ data.cvel[self.base_body_id][:3]).astype(np.float32)

        cmd = np.array(command, dtype=np.float32)

        # Observation frame layout (45 values):
        # [0:3]   command (vx, vy, yaw_rate)
        # [3:6]   gyro * 0.25
        # [6:9]   projected gravity
        # [9:21]  joint position error: q - Q0
        # [21:33] joint velocity * 0.05
        # [33:45] previous raw action
        frame = np.concatenate([
            cmd,
            gyro * 0.25,
            gravity_proj,
            q - Q0,
            dq * 0.05,
            self.previous_action
        ]).astype(np.float32)
        return frame

    def policy_step(self, command: tuple[float, float, float]):
        """Run 50 Hz policy inference and compute target joint positions."""
        frame = self.get_current_frame(command)

        if not self.initialized:
            # Repeat first valid frame 5 times
            self.history = np.tile(frame, (5, 1))
            self.initialized = True
        else:
            # Roll buffer: shift left by 1 and append newest at index 4 (oldest-first)
            self.history[:-1] = self.history[1:]
            self.history[-1] = frame

        # Checkpoint student policy expects [1, 225] oldest-first.
        obs_tensor = torch.from_numpy(self.history.flatten()).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            raw_action = self.policy(obs_tensor).squeeze(0).cpu().numpy().astype(np.float32)

        if not np.all(np.isfinite(raw_action)):
            raw_action = np.zeros(12, dtype=np.float32)

        # Action clipping & scaling
        clipped_action = np.clip(raw_action, -6.0, 6.0)
        self.previous_action = clipped_action.copy()

        # Compute and clamp target positions
        scaled_target = Q0 + SCALE * clipped_action
        self.target_q = np.clip(scaled_target, TARGET_MIN, TARGET_MAX)

    def physics_step(self):
        """Execute one 200 Hz physics step with coupled belt-drive PD torque."""
        q = self.data.qpos[self.qpos_idx].astype(np.float32)
        dq = self.data.qvel[self.qvel_idx].astype(np.float32)

        # Coupled motor-space PD:
        # tau_motor = (kp/gear^2) * (J @ (target - q)) - (kd/gear^2) * (J @ dq)
        # tau_joint = tau_motor @ J^T
        pos_err_motor = J_COUPLING @ (self.target_q - q)
        vel_motor = J_COUPLING @ dq

        tau_motor = (self.kp / (GEAR ** 2)) * pos_err_motor - (self.kd / (GEAR ** 2)) * vel_motor
        tau_joint = tau_motor @ J_COUPLING

        # Clamp to hardware torque limits
        tau_clamped = np.clip(tau_joint, -TORQUE_LIMITS, TORQUE_LIMITS)
        self.data.ctrl[:] = tau_clamped

        mujoco.mj_step(self.model, self.data)
        self.step_count += 1
