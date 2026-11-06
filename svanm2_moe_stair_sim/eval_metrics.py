"""Metric functions for the SvanM2 MoE evaluation battery.

No MuJoCo import here, so these run and test without the simulator.

Definitions follow tools/jitter_analysis/analyze.py:
  * measured joint shaking  = Welch RMS of joint angle, 8-30 Hz, 200 Hz samples
  * requested target shaking = Welch RMS of requested joint angle, 8-20 Hz, 50 Hz ticks
"""

from __future__ import annotations

import math

import numpy as np
from scipy.signal import welch

G = 9.81
JOINT_BAND_HZ = (8.0, 30.0)
TARGET_BAND_HZ = (8.0, 20.0)
TRACK_SIGMA = 0.25            # exp-kernel std for velocity tracking scores [m/s, rad/s]
TOP_X = 3.622                 # 12 treads x 0.30 m + 22 mm foot radius: feet past the last tread
TRACK_HOLD_S = 0.5            # all feet must stay past TOP_X this long
LANE_HALF_WIDTH = 1.0         # stair width 2.0 m
MAX_TILT_TRAVERSE_DEG = 30.0

FLAT_METRIC_KEYS = (
    "steady_s", "vx_mean", "vy_mean", "wz_mean",
    "vx_err_rms", "vy_err_rms", "wz_err_rms",
    "track_lin_score", "track_yaw_score", "speed_mean", "progress_m",
    "power_mean_W", "cot", "torque_sat_frac", "tau_rate_mean", "tau_peak_Nm",
    "joint_band_mean_deg", "joint_band_max_deg", "request_band_mean_deg",
    "request_band_max_deg", "target_limit_frac", "action_rate_mean",
    "action_accel_mean", "tilt_max_deg",
)


def band_rms(values, fs: float, low: float, high: float) -> np.ndarray:
    """Welch band RMS for each column of ``values`` (time along axis 0).

    Returns NaN per column when there is less than two seconds of data.
    """
    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        values = values[:, None]
    if high >= fs / 2:
        raise ValueError("band must lie below the Nyquist frequency")
    if values.shape[0] < round(2 * fs):
        return np.full(values.shape[1], np.nan)
    freq, psd = welch(values, fs=fs, axis=0, nperseg=round(2 * fs), detrend="linear")
    band = (freq >= low) & (freq <= high)
    return np.sqrt(np.trapezoid(psd[band], freq[band], axis=0))


def first_hold(mask, hold_samples: int):
    """Index where ``mask`` first stays True for ``hold_samples`` in a row, else None."""
    streak = 0
    for index, flag in enumerate(mask):
        streak = streak + 1 if flag else 0
        if streak >= hold_samples:
            return index - streak + 1
    return None


def _mean(values) -> float:
    values = np.asarray(values, dtype=float)
    return float(values.mean()) if values.size else math.nan


def _rms(values) -> float:
    values = np.asarray(values, dtype=float)
    return float(np.sqrt(np.mean(values ** 2))) if values.size else math.nan


def flat_metrics(run: dict, cmd, *, transient_s: float, mass_kg: float, dt: float,
                 policy_dt: float, torque_limits, sigma: float = TRACK_SIGMA) -> dict:
    """Summarise one command rollout after discarding ``transient_s`` seconds.

    ``run`` is the dict produced by ``eval_policy_svanm2.run_case``. Arrays are
    physics-rate (``t``, ``q``, ``dq``, ``tau``, ``tau_pre``, ``vx``, ``vy``, ``wz``,
    ``px``, ``py``, ``tilt``) and policy-rate (``tick_*``).
    """
    cx, cy, cw = (float(v) for v in cmd)
    t = run["t"]
    steady = t >= transient_s
    tick_steady = run["tick_t"] >= transient_s
    result = {key: math.nan for key in FLAT_METRIC_KEYS}
    n_joints = run["q"].shape[1] if run["q"].ndim == 2 else 12
    result["_joint_band_deg"] = np.full(n_joints, math.nan)
    result["_request_band_deg"] = np.full(n_joints, math.nan)
    result["steady_s"] = float(steady.sum() * dt)
    result["tilt_max_deg"] = float(np.max(run["tilt"])) if run["tilt"].size else math.nan
    if steady.sum() * dt < 2.0 or not tick_steady.any():
        return result

    vx, vy, wz = run["vx"][steady], run["vy"][steady], run["wz"][steady]
    result.update(
        vx_mean=_mean(vx), vy_mean=_mean(vy), wz_mean=_mean(wz),
        vx_err_rms=_rms(vx - cx), vy_err_rms=_rms(vy - cy), wz_err_rms=_rms(wz - cw),
        track_lin_score=_mean(np.exp(-((vx - cx) ** 2 + (vy - cy) ** 2) / sigma ** 2)),
        track_yaw_score=_mean(np.exp(-((wz - cw) ** 2) / sigma ** 2)),
    )
    speed = float(np.hypot(vx, vy).mean())
    result["speed_mean"] = speed

    tau = run["tau"][steady]
    dq = run["dq"][steady]
    power = np.abs(tau * dq).sum(axis=1)
    result["power_mean_W"] = _mean(power)
    result["cot"] = float(power.mean() / (mass_kg * G * speed)) if speed > 0.05 else math.nan
    result["torque_sat_frac"] = _mean(np.abs(run["tau_pre"][steady]) > torque_limits)
    result["tau_peak_Nm"] = float(np.abs(tau).max())
    tau_rate = np.diff(run["tau"], axis=0) / dt
    result["tau_rate_mean"] = _mean(np.abs(tau_rate[steady[1:]]))

    px, py = run["px"][steady], run["py"][steady]
    disp = np.array([px[-1] - px[0], py[-1] - py[0]])
    direction = np.array([cx, cy])
    norm = float(np.linalg.norm(direction))
    result["progress_m"] = float(disp @ direction / norm) if norm > 1e-6 else math.nan

    joint_band = band_rms(np.degrees(run["q"][steady]), 1.0 / dt, *JOINT_BAND_HZ)
    result["joint_band_mean_deg"] = _mean(joint_band)
    result["joint_band_max_deg"] = float(np.max(joint_band))

    requested = run["tick_requested"][tick_steady]
    target = run["tick_target"][tick_steady]
    request_band = band_rms(np.degrees(requested), 1.0 / policy_dt, *TARGET_BAND_HZ)
    result["request_band_mean_deg"] = _mean(request_band)
    result["request_band_max_deg"] = float(np.max(request_band))
    result["target_limit_frac"] = _mean(np.any(np.abs(requested - target) > 1e-6, axis=1))

    action = run["tick_action"][tick_steady]
    result["action_rate_mean"] = _mean(np.abs(np.diff(action, axis=0)))
    result["action_accel_mean"] = _mean(np.abs(np.diff(action, n=2, axis=0)))
    result["_joint_band_deg"] = joint_band
    result["_request_band_deg"] = request_band
    return result


def stair_metrics(run: dict, track_y: float, *, dt: float) -> dict:
    """Traversal outcome for one stair run.

    Traversed means: all four feet beyond ``TOP_X`` for ``TRACK_HOLD_S``, the base
    stayed in the lane, tilt stayed below ``MAX_TILT_TRAVERSE_DEG``, and no fall.
    Not checked: per-foot platform load (>1 N) and clearance (>0.18 m) from the
    interiit_xterra criteria. Treat this as a first-pass screen, not a final verdict.
    """
    t = run["t"]
    if t.size == 0:
        return {"max_x_m": math.nan, "traversed": False,
                "time_to_top_s": math.nan, "lane_dev_max_m": math.nan}
    beyond = np.all(run["feet_x"] > TOP_X, axis=1)
    index = first_hold(beyond, round(TRACK_HOLD_S / dt))
    lane_dev = float(np.max(np.abs(run["py"] - track_y))) if t.size else math.nan
    tilt_max = float(np.max(run["tilt"])) if t.size else math.nan
    traversed = (
        index is not None
        and not run["fell"]
        and lane_dev < LANE_HALF_WIDTH
        and tilt_max < MAX_TILT_TRAVERSE_DEG
    )
    return {
        "max_x_m": float(np.max(run["px"])) if t.size else math.nan,
        "traversed": bool(traversed),
        "time_to_top_s": float(t[index]) if index is not None else math.nan,
        "lane_dev_max_m": lane_dev,
    }
