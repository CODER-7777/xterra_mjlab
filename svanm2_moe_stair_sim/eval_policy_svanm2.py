#!/usr/bin/env python3
"""Standard evaluation battery for the SvanM2 MoE stair policy (MuJoCo, CPU).

Layout follows interiit_xterra's eval_hs_dhav1w.sh / eval_policy_dhav1w.py:
  pass 1  flat command battery: forward, lateral and yaw sweeps on a floor-only scene
  pass 2  stairs: one traversal per track, optional start offsets

Each run starts from a fresh MuJoCo state and uses the same policy, observation
and torque law as the interactive viewer (controller.py). Outputs under --out:
  runs.csv, summary.csv, manifest.json, flat_battery.png, stairs.png,
  and TensorBoard scalars in tensorboard/ (disable with --no-tensorboard).

Examples, from Task_3:
  xterra_mjlab/.venv/bin/python xterra_mjlab/svanm2_moe_stair_sim/eval_policy_svanm2.py
  xterra_mjlab/.venv/bin/python xterra_mjlab/svanm2_moe_stair_sim/eval_policy_svanm2.py \
      --no-stairs --contact-impratio 10 --label impratio10
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from build_scene import MESH_DIR, TRACK_CONFIGS, get_spawn_pos, get_track_y  # noqa: E402
from checkpoint_policy import DEFAULT_CHECKPOINT  # noqa: E402
from controller import GEAR, J_COUPLING, JOINTS, Q0, SCALE, TORQUE_LIMITS, MoeStairController  # noqa: E402
from eval_metrics import FLAT_METRIC_KEYS, TOP_X, flat_metrics, stair_metrics  # noqa: E402

TASK3_DIR = HERE.parents[1]
SCENE_XML = HERE / "scene.xml"
POLICY_DECIMATION = 4                       # 50 Hz policy from 200 Hz physics
FOOT_GEOMS = ("FL_foot", "FR_foot", "RL_foot", "RR_foot")
FALL_TILT_DEG = 60.0
FALL_HEIGHT_M = 0.15
SPAWN_Z = 0.365                             # same standing height as build_scene
STAIR_STARTS = {"center": (0.0, 0.0), "minus": (-0.03, -0.02), "plus": (0.03, 0.02)}


def build_flat_scene(out_dir: Path) -> Path:
    """Copy scene.xml with the six stair tracks removed (floor and robot kept)."""
    root = ET.parse(SCENE_XML).getroot()
    worldbody = root.find("worldbody")
    compiler = root.find("compiler")
    if worldbody is None or compiler is None:
        raise RuntimeError("scene.xml lacks worldbody or compiler")
    tracks = [b for b in worldbody.findall("body") if b.get("name", "").startswith("track_")]
    if len(tracks) != len(TRACK_CONFIGS):
        raise RuntimeError(f"expected {len(TRACK_CONFIGS)} track bodies, found {len(tracks)}")
    for body in tracks:
        worldbody.remove(body)
    compiler.set("meshdir", str(MESH_DIR))      # absolute, so the copy can live elsewhere
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "flat_scene.xml"
    ET.ElementTree(root).write(path)
    return path


def robot_mass(model: mujoco.MjModel, base_id: int) -> float:
    """Mass of the base body and everything attached below it (excludes static tracks)."""
    total = 0.0
    for body in range(1, model.nbody):
        node = body
        while node > 0 and node != base_id:
            node = int(model.body_parentid[node])
        if node == base_id:
            total += float(model.body_mass[body])
    return total


def pre_clip_torque(ctrl: MoeStairController, data: mujoco.MjData) -> np.ndarray:
    """Joint torque before the torque clamp, mirroring controller.physics_step."""
    q = data.qpos[ctrl.qpos_idx].astype(np.float32)
    dq = data.qvel[ctrl.qvel_idx].astype(np.float32)
    pos_err_motor = J_COUPLING @ (ctrl.target_q - q)
    vel_motor = J_COUPLING @ dq
    tau_motor = (ctrl.kp / GEAR ** 2) * pos_err_motor - (ctrl.kd / GEAR ** 2) * vel_motor
    return (tau_motor @ J_COUPLING).astype(np.float64)


def base_origin_velocity_body(data: mujoco.MjData, base_id: int, rot: np.ndarray) -> np.ndarray:
    """Body-frame linear velocity of the base origin.

    data.cvel's linear part is the velocity of the subtree centre of mass, not of the
    base origin. Removing w x (subtree_com - xpos) gives the origin velocity, as in
    interiit_xterra's root_link_lin_vel_b. Without this, a pitching base reads a
    phantom velocity of the order of w x r.
    """
    cvel = data.cvel[base_id]
    offset = data.subtree_com[base_id] - data.xpos[base_id]
    lin_origin = cvel[3:6] - np.cross(cvel[:3], offset)
    return rot.T @ lin_origin


def run_case(model, data, ctrl, foot_ids, base_id, cmd, duration, spawn, yaw) -> dict:
    """Roll one command for ``duration`` seconds. Stops early on a fall."""
    mujoco.mj_resetData(model, data)
    ctrl.reset(target_pos=spawn, yaw=yaw)
    n_steps = int(round(duration / model.opt.timestep))
    phys = {k: [] for k in ("t", "q", "dq", "tau", "tau_pre", "vx", "vy", "wz",
                            "px", "py", "pz", "tilt", "feet_x")}
    ticks = {k: [] for k in ("t", "requested", "target", "action")}
    fell = False
    for step in range(n_steps):
        rot = data.xmat[base_id].reshape(3, 3)
        tilt = float(np.degrees(np.arccos(np.clip(rot[2, 2], -1.0, 1.0))))
        if tilt > FALL_TILT_DEG or data.xpos[base_id][2] < FALL_HEIGHT_M:
            fell = True
            break
        if step % POLICY_DECIMATION == 0:
            ctrl.policy_step(command=cmd)
            ticks["t"].append(data.time)
            ticks["requested"].append(Q0 + SCALE * ctrl.previous_action)
            ticks["target"].append(ctrl.target_q.copy())
            ticks["action"].append(ctrl.previous_action.copy())
        lin_b = base_origin_velocity_body(data, base_id, rot)
        ang_b = rot.T @ data.cvel[base_id][:3]
        phys["t"].append(data.time)
        phys["q"].append(data.qpos[ctrl.qpos_idx].copy())
        phys["dq"].append(data.qvel[ctrl.qvel_idx].copy())
        phys["tau_pre"].append(pre_clip_torque(ctrl, data))
        phys["vx"].append(lin_b[0])
        phys["vy"].append(lin_b[1])
        phys["wz"].append(ang_b[2])
        phys["px"].append(data.xpos[base_id][0])
        phys["py"].append(data.xpos[base_id][1])
        phys["pz"].append(data.xpos[base_id][2])
        phys["tilt"].append(tilt)
        phys["feet_x"].append(data.geom_xpos[foot_ids, 0].copy())
        ctrl.physics_step()
        phys["tau"].append(data.ctrl[:12].copy())
    out = {k: np.asarray(v, dtype=float) for k, v in phys.items()}
    out.update({f"tick_{k}": np.asarray(v, dtype=float) for k, v in ticks.items()})
    out["fell"] = fell
    return out


def flat_battery(vx_vals, vy_vals, wz_vals) -> list[dict]:
    runs = [{"section": "flat", "axis": "forward", "cmd": (v, 0.0, 0.0), "track": "", "start": ""}
            for v in vx_vals]
    # Zero command is identical on every axis, so it is run once, on the forward axis.
    runs += [{"section": "flat", "axis": "lateral", "cmd": (0.0, v, 0.0), "track": "", "start": ""}
             for v in vy_vals if v != 0.0]
    runs += [{"section": "flat", "axis": "yaw", "cmd": (0.0, 0.0, v), "track": "", "start": ""}
             for v in wz_vals if v != 0.0]
    return runs


def stair_battery(tracks, starts, vx) -> list[dict]:
    runs = []
    for track in tracks:
        for name in starts:
            runs.append({"section": "stairs", "axis": "forward", "cmd": (vx, 0.0, 0.0),
                         "track": int(track), "start": name})
    return runs


def load_controller(scene_path: Path, args, impratio: float):
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    model.opt.impratio = float(impratio)
    data = mujoco.MjData(model)
    ctrl = MoeStairController(model, data, checkpoint_path=args.checkpoint_file,
                              device=args.device, observation_timing=args.observation_timing)
    foot_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name) for name in FOOT_GEOMS]
    base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")
    return model, data, ctrl, np.array(foot_ids), base_id


def evaluate(entries, model, data, ctrl, foot_ids, base_id, *, duration, transient, args):
    """Run every battery entry and return one flat row per entry."""
    mass = robot_mass(model, base_id)
    dt = float(model.opt.timestep)
    policy_dt = dt * POLICY_DECIMATION
    rows = []
    for index, entry in enumerate(entries, start=1):
        cmd = entry["cmd"]
        if entry["section"] == "flat":
            spawn, yaw, dur = (0.0, 0.0, SPAWN_Z), 0.0, duration
        else:
            base_x, base_y, base_z = get_spawn_pos(entry["track"])
            dy, dyaw = STAIR_STARTS[entry["start"]]
            spawn, yaw, dur = (base_x, base_y + dy, base_z), dyaw, args.stair_duration
        started = time.perf_counter()
        run = run_case(model, data, ctrl, foot_ids, base_id, cmd, dur, spawn, yaw)
        metrics = flat_metrics(run, cmd, transient_s=transient, mass_kg=mass, dt=dt,
                               policy_dt=policy_dt, torque_limits=TORQUE_LIMITS)
        row = {
            "section": entry["section"], "axis": entry["axis"],
            "cmd_vx": cmd[0], "cmd_vy": cmd[1], "cmd_wz": cmd[2],
            "track": entry["track"], "start": entry["start"],
            "fell": run["fell"], "duration_s": float(run["t"][-1] - run["t"][0]) if run["t"].size > 1 else 0.0,
        }
        if entry["section"] == "stairs":
            row.update(stair_metrics(run, get_track_y(entry["track"] - 1), dt=dt))
        else:
            row.update(traversed="", max_x_m="", time_to_top_s="", lane_dev_max_m="")
        for key in FLAT_METRIC_KEYS:
            row[key] = metrics[key]
        for name, value in zip(JOINTS, metrics["_joint_band_deg"]):
            row[f"joint_band_deg_{name}"] = float(value)
        for name, value in zip(JOINTS, metrics["_request_band_deg"]):
            row[f"request_band_deg_{name}"] = float(value)
        row["wall_s"] = time.perf_counter() - started
        rows.append(row)
        label = (f"{entry['axis']} cmd={cmd}" if entry["section"] == "flat"
                 else f"track {entry['track']} start={entry['start']}")
        print(f"[eval] {index:3d}/{len(entries)} {label}: fell={run['fell']} "
              f"joint_band={row['joint_band_mean_deg']:.3f} deg  wall={row['wall_s']:.1f}s", flush=True)
    return rows


def write_csv(rows, path: Path):
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _group_mean(rows, key):
    values = [r[key] for r in rows if isinstance(r.get(key), (int, float)) and not isinstance(r.get(key), bool)]
    values = [v for v in values if np.isfinite(v)]
    return float(np.mean(values)) if values else float("nan")


def _group_max(rows, key):
    values = [r[key] for r in rows if np.isfinite(r[key])]
    return float(np.max(values)) if values else float("nan")


def summary_rows(rows) -> list[dict]:
    groups: dict[str, list[dict]] = {}
    for row in rows:
        name = row["axis"] if row["section"] == "flat" else f"stairs_track{row['track']}"
        groups.setdefault(name, []).append(row)
    out = []
    for name, members in groups.items():
        item = {"group": name, "runs": len(members), "falls": sum(bool(r["fell"]) for r in members)}
        if members[0]["section"] == "flat":
            item.update(
                track_lin_score=_group_mean(members, "track_lin_score"),
                track_yaw_score=_group_mean(members, "track_yaw_score"),
                joint_band_mean_deg=_group_mean(members, "joint_band_mean_deg"),
                joint_band_max_deg=_group_max(members, "joint_band_max_deg"),
                request_band_mean_deg=_group_mean(members, "request_band_mean_deg"),
                target_limit_frac=_group_mean(members, "target_limit_frac"),
                torque_sat_frac=_group_mean(members, "torque_sat_frac"),
                cot=_group_mean(members, "cot"),
            )
        else:
            item.update(
                traversed=sum(bool(r["traversed"]) for r in members),
                time_to_top_s=_group_mean(members, "time_to_top_s"),
                joint_band_mean_deg=_group_mean(members, "joint_band_mean_deg"),
                request_band_mean_deg=_group_mean(members, "request_band_mean_deg"),
            )
        out.append(item)
    return out


def write_summary(rows, path: Path):
    items = summary_rows(rows)
    fieldnames = []
    for item in items:
        for key in item:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, restval="")
        writer.writeheader()
        writer.writerows(items)


def make_figures(rows, out: Path):
    import matplotlib.pyplot as plt

    flat = [r for r in rows if r["section"] == "flat"]
    specs = [("forward", "cmd_vx", "vx_mean", "forward command [m/s]"),
             ("lateral", "cmd_vy", "vy_mean", "lateral command [m/s]"),
             ("yaw", "cmd_wz", "wz_mean", "yaw command [rad/s]")]
    fig, axes = plt.subplots(3, 2, figsize=(11, 11))
    for i, (axis, cmd_key, mean_key, xlabel) in enumerate(specs):
        members = sorted((r for r in flat if r["axis"] == axis), key=lambda r: r[cmd_key])
        if not members:
            continue
        x = np.array([r[cmd_key] for r in members], dtype=float)
        y = np.array([r[mean_key] for r in members], dtype=float)
        left, right = axes[i, 0], axes[i, 1]
        left.plot(x, y, "o-", label="achieved mean")
        left.plot(x, x, "--", color="grey", label="commanded")
        n_fell = sum(bool(r["fell"]) for r in members)
        if n_fell:
            left.set_title(f"{n_fell} run(s) fell", fontsize=9, color="red")
        left.set_xlabel(xlabel)
        left.set_ylabel("body velocity [m/s or rad/s]")
        left.legend(fontsize=8)
        right.plot(x, [r["joint_band_mean_deg"] for r in members], "o-", label="joint shaking 8–30 Hz (measured)")
        right.plot(x, [r["request_band_mean_deg"] for r in members], "s-", label="requested target 8–20 Hz")
        right.set_xlabel(xlabel)
        right.set_ylabel("RMS [deg]")
        right.legend(fontsize=8)
    fig.suptitle("Flat command battery")
    fig.tight_layout()
    fig.savefig(out / "flat_battery.png", dpi=110)
    plt.close(fig)

    stairs = [r for r in rows if r["section"] == "stairs"]
    if stairs:
        labels = [f"{r['track']}·{r['start']}" for r in stairs]
        max_x = [r["max_x_m"] for r in stairs]
        colors = ["tab:green" if r["traversed"] else "tab:red" for r in stairs]
        fig, ax = plt.subplots(figsize=(max(7, 0.6 * len(stairs)), 4))
        ax.bar(labels, max_x, color=colors)
        ax.axhline(TOP_X, linestyle="--", color="black", label="traversal line")
        ax.set_ylabel("max base x [m]")
        ax.set_xlabel("track · start")
        ax.set_title("Stair runs (green = traversed by the screen criteria)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / "stairs.png", dpi=110)
        plt.close(fig)


def write_tensorboard(rows, out: Path):
    from torch.utils.tensorboard import SummaryWriter

    keys = ("joint_band_mean_deg", "request_band_mean_deg", "track_lin_score", "track_yaw_score",
            "cot", "torque_sat_frac", "target_limit_frac", "fell")
    writer = SummaryWriter(log_dir=str(out / "tensorboard"))
    for step, row in enumerate(rows):
        group = row["axis"] if row["section"] == "flat" else f"track{row['track']}_{row['start']}"
        prefix = f"{row['section']}/{group}"
        for key in keys:
            value = row.get(key)
            if isinstance(value, (bool, np.bool_)):
                value = float(value)
            if isinstance(value, (int, float)) and np.isfinite(value):
                writer.add_scalar(f"{prefix}/{key}", float(value), step)
    writer.close()


def checkpoint_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint-file", type=Path, default=DEFAULT_CHECKPOINT)
    ap.add_argument("--device", default="cpu", help="policy device: cpu (default) or cuda:0")
    ap.add_argument("--observation-timing", choices=("legacy", "fresh"), default="legacy")
    ap.add_argument("--contact-impratio", type=float, default=100.0,
                    help="MuJoCo impratio for this evaluation (viewer default 100)")
    ap.add_argument("--label", default=None, help="suffix for the default output folder")
    ap.add_argument("--out", type=Path, default=None,
                    help="output folder (default tools/results/svanm2_eval/<checkpoint>_<label>)")
    ap.add_argument("--no-flat", action="store_true", help="skip the flat command battery")
    ap.add_argument("--no-stairs", action="store_true", help="skip the stair traversals")
    ap.add_argument("--no-tensorboard", action="store_true", help="do not write TensorBoard scalars")
    ap.add_argument("--vx", type=float, nargs="+", default=[-0.5, -0.25, 0.0, 0.25, 0.5],
                    help="forward commands [m/s]; default is the viewer's teleop cap (0.5)")
    ap.add_argument("--vy", type=float, nargs="+", default=[-0.6, -0.3, 0.0, 0.3, 0.6],
                    help="lateral commands [m/s]; default is the viewer's teleop cap (0.6)")
    ap.add_argument("--wz", type=float, nargs="+", default=[-0.4, -0.2, 0.0, 0.2, 0.4],
                    help="yaw-rate commands [rad/s]; default is the viewer's teleop cap (0.4)")
    ap.add_argument("--flat-duration", type=float, default=10.0, help="seconds per flat command")
    ap.add_argument("--flat-transient", type=float, default=3.0,
                    help="seconds discarded before steady-state metrics (stand + command settling)")
    ap.add_argument("--tracks", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6], choices=[1, 2, 3, 4, 5, 6])
    ap.add_argument("--stair-starts", nargs="+", default=["center"], choices=list(STAIR_STARTS),
                    help="start offsets: center; minus = y-0.03 m & yaw-0.02 rad; plus = the opposite")
    ap.add_argument("--stair-vx", type=float, default=0.3, help="forward command on stairs [m/s]")
    ap.add_argument("--stair-duration", type=float, default=30.0, help="seconds per stair run")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    ckpt = args.checkpoint_file.resolve()
    if not ckpt.exists():
        raise SystemExit(f"no such checkpoint: {ckpt}")
    label = args.label or f"impratio{args.contact_impratio:g}"
    out = args.out or (TASK3_DIR / "tools/results/svanm2_eval" / f"{ckpt.stem}_{label}")
    out = out.resolve()
    out.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    if not args.no_flat:
        scene = build_flat_scene(out / "scene")
        model, data, ctrl, foot_ids, base_id = load_controller(scene, args, args.contact_impratio)
        entries = flat_battery(args.vx, args.vy, args.wz)
        print(f"[eval] flat battery: {len(entries)} commands, {args.flat_duration:g} s each", flush=True)
        rows += evaluate(entries, model, data, ctrl, foot_ids, base_id,
                         duration=args.flat_duration, transient=args.flat_transient, args=args)
    if not args.no_stairs:
        model, data, ctrl, foot_ids, base_id = load_controller(SCENE_XML, args, args.contact_impratio)
        entries = stair_battery(args.tracks, args.stair_starts, args.stair_vx)
        print(f"[eval] stair battery: {len(entries)} runs, {args.stair_duration:g} s each", flush=True)
        rows += evaluate(entries, model, data, ctrl, foot_ids, base_id,
                         duration=args.stair_duration, transient=args.flat_transient, args=args)
    if not rows:
        raise SystemExit("nothing to evaluate: both --no-flat and --no-stairs were given")

    write_csv(rows, out / "runs.csv")
    write_summary(rows, out / "summary.csv")
    make_figures(rows, out)
    if not args.no_tensorboard:
        write_tensorboard(rows, out)
    manifest = {
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "checkpoint": str(ckpt),
        "checkpoint_sha256": checkpoint_sha256(ckpt),
        "checkpoint_update": int(ctrl.checkpoint_update),
        "contact_impratio": args.contact_impratio,
        "observation_timing": args.observation_timing,
        "device": args.device,
        "physics_timestep_s": float(model.opt.timestep),
        "policy_decimation": POLICY_DECIMATION,
        "python": platform.python_version(),
        "mujoco": mujoco.__version__,
        "arguments": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "runs": len(rows),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    print(f"[eval] wrote {out / 'runs.csv'}")
    print(f"[eval] summary: {out / 'summary.csv'}")
    if not args.no_tensorboard:
        print(f"[eval] tensorboard: tools/launch_tensorboard.sh {out}")
    return out


if __name__ == "__main__":
    main()
