"""Checks for the SvanM2 evaluation battery: signal maths, traversal logic and scene build.

Run from this folder:
    ../.venv/bin/python -m unittest test_eval_svanm2 -v
"""

import math
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import eval_metrics as M  # noqa: E402
import eval_policy_svanm2 as E  # noqa: E402


def synthetic_run(n_phys, dt, *, vx=0.3, feet_x=0.0, py=0.0, fell=False, n_joints=12):
    """A constant-velocity run with stationary joints, shaped like run_case() output."""
    t = np.arange(n_phys) * dt
    n_ticks = (n_phys + 3) // 4
    ones = np.ones(n_phys)
    return {
        "t": t,
        "q": np.zeros((n_phys, n_joints)),
        "dq": np.zeros((n_phys, n_joints)),
        "tau": np.full((n_phys, n_joints), 0.5),
        "tau_pre": np.full((n_phys, n_joints), 0.5),
        "vx": vx * ones, "vy": np.zeros(n_phys), "wz": np.zeros(n_phys),
        "px": vx * t, "py": py * ones, "pz": 0.365 * ones,
        "tilt": np.zeros(n_phys),
        "feet_x": np.full((n_phys, 4), feet_x),
        "tick_t": np.arange(n_ticks) * 4 * dt,
        "tick_requested": np.zeros((n_ticks, n_joints)),
        "tick_target": np.zeros((n_ticks, n_joints)),
        "tick_action": np.zeros((n_ticks, n_joints)),
        "fell": fell,
    }


class BandRmsTests(unittest.TestCase):
    def test_in_band_sine_has_rms_of_its_amplitude(self):
        fs, n = 200.0, 2000
        t = np.arange(n) / fs
        x = np.sin(2 * np.pi * 12.0 * t)          # amplitude 1 -> RMS 1/sqrt(2)
        value = M.band_rms(x, fs, 8.0, 30.0)[0]
        self.assertAlmostEqual(value, 1 / math.sqrt(2), delta=0.05)

    def test_out_of_band_sine_is_rejected(self):
        fs, n = 200.0, 2000
        t = np.arange(n) / fs
        x = np.sin(2 * np.pi * 60.0 * t)          # above the 8-30 Hz band
        self.assertLess(M.band_rms(x, fs, 8.0, 30.0)[0], 0.05)

    def test_short_record_is_nan_not_zero(self):
        self.assertTrue(np.isnan(M.band_rms(np.ones(100), 200.0, 8.0, 30.0)).all())

    def test_band_above_nyquist_raises(self):
        with self.assertRaises(ValueError):
            M.band_rms(np.ones(1000), 50.0, 8.0, 30.0)


class HoldTests(unittest.TestCase):
    def test_first_hold_requires_consecutive_samples(self):
        mask = np.array([1, 1, 0, 1, 1, 1, 1, 0], dtype=bool)
        self.assertEqual(M.first_hold(mask, 4), 3)
        self.assertIsNone(M.first_hold(mask, 5))


class StairMetricTests(unittest.TestCase):
    dt = 0.005

    def test_feet_past_top_and_in_lane_traverse(self):
        run = synthetic_run(2000, self.dt, feet_x=M.TOP_X + 0.1)
        result = M.stair_metrics(run, track_y=0.0, dt=self.dt)
        self.assertTrue(result["traversed"])
        self.assertAlmostEqual(result["time_to_top_s"], 0.0)

    def test_short_hold_does_not_traverse(self):
        run = synthetic_run(2000, self.dt, feet_x=M.TOP_X + 0.1)
        run["feet_x"][::2] = 0.0                  # never held for TRACK_HOLD_S
        self.assertFalse(M.stair_metrics(run, track_y=0.0, dt=self.dt)["traversed"])

    def test_leaving_the_lane_or_falling_fails(self):
        out_of_lane = synthetic_run(2000, self.dt, feet_x=M.TOP_X + 0.1, py=1.5)
        self.assertFalse(M.stair_metrics(out_of_lane, track_y=0.0, dt=self.dt)["traversed"])
        fell = synthetic_run(2000, self.dt, feet_x=M.TOP_X + 0.1, fell=True)
        self.assertFalse(M.stair_metrics(fell, track_y=0.0, dt=self.dt)["traversed"])

    def test_empty_run_is_safe(self):
        run = synthetic_run(0, self.dt)
        result = M.stair_metrics(run, track_y=0.0, dt=self.dt)
        self.assertFalse(result["traversed"])
        self.assertTrue(math.isnan(result["max_x_m"]))


class FlatMetricTests(unittest.TestCase):
    dt = 0.005
    policy_dt = 0.02

    def metrics(self, run, cmd):
        return M.flat_metrics(run, cmd, transient_s=1.0, mass_kg=10.0, dt=self.dt,
                              policy_dt=self.policy_dt, torque_limits=np.full(12, 12.0))

    def test_perfect_tracking_scores_one_and_still_joints_have_no_shaking(self):
        result = self.metrics(synthetic_run(2000, self.dt, vx=0.3), (0.3, 0.0, 0.0))
        self.assertAlmostEqual(result["track_lin_score"], 1.0, places=6)
        self.assertAlmostEqual(result["vx_err_rms"], 0.0, places=9)
        self.assertLess(result["joint_band_mean_deg"], 1e-9)
        self.assertAlmostEqual(result["target_limit_frac"], 0.0)
        self.assertFalse(math.isnan(result["cot"]))

    def test_short_fall_gives_nan_metrics_and_full_length_band_arrays(self):
        result = self.metrics(synthetic_run(100, self.dt, fell=True), (0.3, 0.0, 0.0))
        self.assertTrue(math.isnan(result["track_lin_score"]))
        self.assertEqual(result["_joint_band_deg"].shape, (12,))
        self.assertTrue(np.isnan(result["_joint_band_deg"]).all())


class BatteryTests(unittest.TestCase):
    def test_zero_command_is_run_once(self):
        runs = E.flat_battery([0.0, 0.25], [0.0, 0.3], [0.0])
        commands = [r["cmd"] for r in runs]
        self.assertEqual(commands.count((0.0, 0.0, 0.0)), 1)
        self.assertEqual(len(runs), 3)

    def test_stair_battery_covers_each_track_and_start(self):
        runs = E.stair_battery([1, 6], ["center", "plus"], 0.3)
        self.assertEqual(len(runs), 4)
        self.assertEqual({(r["track"], r["start"]) for r in runs},
                         {(1, "center"), (1, "plus"), (6, "center"), (6, "plus")})


class BaseVelocityTests(unittest.TestCase):
    def test_origin_velocity_removes_rotation_about_offset_com(self):
        # Pure yaw rate 1 rad/s about z, centre of mass 1 m along +x from the base origin.
        # The origin then moves at -y: v_origin = w x (origin - com) = (0, -1, 0).
        data = SimpleNamespace(
            cvel=np.array([[0.0, 0.0, 1.0, 0.0, 0.0, 0.0]]),
            subtree_com=np.array([[1.0, 0.0, 0.0]]),
            xpos=np.array([[0.0, 0.0, 0.0]]),
        )
        v = E.base_origin_velocity_body(data, 0, np.eye(3))
        np.testing.assert_allclose(v, [0.0, -1.0, 0.0], atol=1e-12)

    def test_no_correction_when_com_is_at_origin(self):
        data = SimpleNamespace(
            cvel=np.array([[0.0, 0.0, 2.0, 0.3, 0.0, 0.0]]),
            subtree_com=np.array([[0.0, 0.0, 0.0]]),
            xpos=np.array([[0.0, 0.0, 0.0]]),
        )
        np.testing.assert_allclose(E.base_origin_velocity_body(data, 0, np.eye(3)), [0.3, 0.0, 0.0])


class SceneTests(unittest.TestCase):
    def test_flat_scene_drops_tracks_and_still_loads(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = E.build_flat_scene(Path(tmp))
            worldbody = ET.parse(path).getroot().find("worldbody")
            tree_names = [b.get("name", "") for b in worldbody.findall("body")]
            self.assertFalse([n for n in tree_names if n.startswith("track_")])
            model = mujoco.MjModel.from_xml_path(str(path))
            base = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")
            self.assertGreater(base, 0)
            # Only the robot carries mass; the floor is a plane with no body mass.
            self.assertAlmostEqual(E.robot_mass(model, base), float(model.body_mass.sum()), places=6)


class SummaryTests(unittest.TestCase):
    def test_flat_and_stairs_group_separately(self):
        rows = [
            {"section": "flat", "axis": "forward", "track": "", "fell": False,
             "track_lin_score": 0.9, "track_yaw_score": 1.0, "joint_band_mean_deg": 0.3,
             "joint_band_max_deg": 0.5, "request_band_mean_deg": 1.0, "target_limit_frac": 0.0,
             "torque_sat_frac": 0.0, "cot": 0.6},
            {"section": "stairs", "axis": "forward", "track": 1, "start": "center", "fell": True,
             "traversed": False, "time_to_top_s": math.nan, "joint_band_mean_deg": 0.8,
             "request_band_mean_deg": 2.0},
        ]
        groups = {item["group"]: item for item in E.summary_rows(rows)}
        self.assertEqual(groups["forward"]["runs"], 1)
        self.assertEqual(groups["stairs_track1"]["falls"], 1)
        self.assertEqual(groups["stairs_track1"]["traversed"], 0)


if __name__ == "__main__":
    unittest.main()
