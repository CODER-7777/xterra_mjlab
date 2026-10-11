"""Interactive MuJoCo simulation runner for SvanM2 with MoE 90k policy on variable stairs.

Features:
- Real-time MuJoCo passive viewer with tracking camera
- Dual teleoperation: non-blocking terminal WASD and viewer arrow keys
- Automatic protection against MuJoCo hotkey toggles (wireframe/actuators)
- 6 distinct stair tracks (10cm, 12cm, 14cm, 15cm, 16cm, 20cm x 12 steps each)
- Turnaround platforms at top of stairs
- Instant track teleport keys (1-6) and top platform teleport (T)
- Viewport and terminal HUDs
"""

import argparse
import sys
import time
from pathlib import Path
import numpy as np
import mujoco
import mujoco.viewer

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from build_scene import build_scene_xml, TRACK_CONFIGS, TRACK_SPACING
from controller import MoeStairController
from checkpoint_policy import DEFAULT_CHECKPOINT
from teleop import KeyboardTeleop, MAX_VX, MAX_VY, MAX_YAW


def positive_finite(value):
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError("Contact impedance ratio must be positive and finite")
    return value


def configure_contact(model, impratio=None):
    """Explicit simulation-only intervention; preserve the XML default if omitted."""
    if impratio is not None:
        model.opt.impratio = positive_finite(impratio)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="SvanM2 MoE 90k Interactive Stair Simulation")
    parser.add_argument("--checkpoint-file", type=Path, default=DEFAULT_CHECKPOINT,
                        help="MoE-CTS training checkpoint (default: checkpoints/svanm2_moe/model_90000.pt)")
    parser.add_argument("--device", default="cuda:0",
                        help="Policy inference device: cuda:0 (default) or cpu; MuJoCo physics remains on CPU")
    parser.add_argument("--track", type=int, default=1, choices=[1, 2, 3, 4, 5, 6],
                        help="Initial track number (1: 10cm, 2: 12cm, 3: 14cm, 4: 15cm, 5: 16cm, 6: 20cm)")
    parser.add_argument("--top", action="store_true", help="Spawn directly on the top platform")
    parser.add_argument("--rebuild-scene", action="store_true", help="Force rebuild of scene.xml")
    parser.add_argument("--record", action="store_true", help="Log your actual interactive run to tools/results/interactive/<timestamp>")
    parser.add_argument("--observation-timing", choices=("legacy", "fresh"), default="legacy",
                        help="Legacy readings or consistent post-step readings refreshed on copied state")
    parser.add_argument("--log-dir", type=Path, help="Record to this new directory instead of an automatic timestamp")
    parser.add_argument("--contact-impratio", type=positive_finite, default=None,
                        help="Explicit MuJoCo contact experiment (10 reduced measured jitter); omitted uses scene.xml. Not a hardware parameter.")
    return parser.parse_args(argv)


def print_hud(info: dict, teleop: KeyboardTeleop, data: mujoco.MjData, tilt_deg: float, sim_time: float):
    """Print clean terminal HUD overlay."""
    pos = data.qpos[:3]
    print(f"\r[t={sim_time:5.1f}s] "
          f"Track {info['track']} ({info['label']}, 12 steps) | "
          f"{info['region']:<25} | "
          f"Pos: [{pos[0]:+5.2f}, {pos[1]:+5.2f}, {pos[2]:4.2f}] | "
          f"Tilt: {tilt_deg:4.1f}° | "
          f"Cmd: [vx={teleop.vx:+.2f}, vy={teleop.vy:+.2f}, yaw={teleop.yaw_rate:+.2f}]  ",
          end="", flush=True)


def main():
    args = parse_args()

    scene_path = HERE / "scene.xml"
    if args.rebuild_scene or not scene_path.exists():
        print("[INFO] Building multi-stair scene XML...")
        build_scene_xml(scene_path)

    print("[INFO] Loading MuJoCo model...")
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    configure_contact(model, args.contact_impratio)
    print(f"[INFO] Active contact impratio: {model.opt.impratio:g}; physics timestep: {model.opt.timestep:g} s")
    data = mujoco.MjData(model)

    print("[INFO] Initializing MoE 90k Controller and Teleop...")
    ctrl = MoeStairController(model, data, checkpoint_path=args.checkpoint_file, device=args.device,
                             observation_timing=args.observation_timing)
    session = None
    if args.record or args.log_dir:
        sys.path.insert(0, str(HERE.parents[1] / "tools/jitter_analysis"))
        from interactive import InteractiveSession, RecordedTeleop
        teleop = RecordedTeleop(ctrl, TRACK_CONFIGS, track_spacing=TRACK_SPACING)
    else:
        teleop = KeyboardTeleop(ctrl, TRACK_CONFIGS, track_spacing=TRACK_SPACING)

    # Spawn at requested track
    teleop.teleport_to_track(args.track, on_top=args.top)
    if args.record or args.log_dir:
        session = InteractiveSession(model, data, ctrl, teleop, args.log_dir)

    print("\n" + "="*80)
    print(f" SVAN M2 — MOE UPDATE {ctrl.checkpoint_update} MULTI-STAIR INTERACTIVE SIMULATION")
    print("="*80)
    print(" Tracks: [1] 10cm  [2] 12cm  [3] 14cm  [4] 15cm  [5] 16cm  [6] 20cm (12 steps each)")
    print("\n CONTROLS:")
    print(" 1. In Terminal Console (Recommended — zero viewer hotkey interference):")
    print("    [W] / [S]        : Increase / Decrease forward velocity (max ±0.50 m/s)")
    print("    [A] / [D]        : Strafe Left / Right (max ±0.60 m/s)")
    print("    [Q] / [E]        : Turn Left / Right (max ±0.40 rad/s)")
    print("    [Space]          : STOP (Zero all velocities)")
    print("    [1] - [6]        : Teleport to approach runway of Track 1 to 6")
    print("    [T]              : Teleport to TOP PLATFORM of current track (face downstairs)")
    print("    [R]              : Reset robot at current track entrance")
    print("\n 2. In MuJoCo Viewer Window:")
    print("    [Up] / [Down]    : Forward / Backward")
    print("    [Left] / [Right] : Strafe Left / Right")
    print("    [I] / [K]        : Forward / Backward")
    print("    [J] / [U]        : Strafe Left / Right")
    print("    [Space]          : STOP")
    print("="*80 + "\n")

    decimation = 4  # 50 Hz policy from 200 Hz physics
    dt = model.opt.timestep  # 0.005 s

    # Start non-blocking terminal input listener
    teleop.start_terminal_listener()

    error = None
    try:
        # Launch passive viewer
        with mujoco.viewer.launch_passive(model, data, key_callback=teleop.handle_key) as viewer:
            # Camera tracking robot base
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            viewer.cam.trackbodyid = ctrl.base_body_id
            viewer.cam.distance = 2.80
            viewer.cam.elevation = -18.0
            viewer.cam.azimuth = 45.0

            step = 0
            last_hud_time = 0.0
            wall_start = time.time()
            sim_start = data.time

            while viewer.is_running():
                if session:
                    teleop.apply_pending(session)
                # Prevent MuJoCo hotkey toggles from corrupting the visual rendering
                viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_ACTUATOR] = 0
                viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_LIGHT] = 0
                viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_SKIN] = 0
                viewer.opt.label = mujoco.mjtLabel.mjLABEL_NONE
                if hasattr(viewer, 'user_scn') and viewer.user_scn is not None:
                    viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_WIREFRAME] = 0

                # Policy step at 50 Hz
                policy_tick = step % decimation == 0
                command = teleop.command
                if session:
                    session.prepare_policy(command, policy_tick)
                if policy_tick:
                    ctrl.policy_step(command=command)

                # Physics step at 200 Hz
                if session:
                    session.before_physics(command, teleop.current_track)
                ctrl.physics_step()
                if session:
                    session.after_physics()
                step += 1

                # Calculate base tilt
                rot = data.xmat[ctrl.base_body_id].reshape(3, 3)
                tilt_deg = float(np.degrees(np.arccos(np.clip(rot[2, 2], -1.0, 1.0))))

                # Update viewer at ~30-50 Hz
                if step % decimation == 0:
                    # In-viewport overlay HUD
                    if step % (decimation * 5) == 0:
                        info = teleop.get_track_info(float(data.qpos[1]), float(data.qpos[0]))
                        col1 = f"Track: {info['track']} ({info['label']})\n{info['region']}\nCmd: vx={teleop.vx:+.2f}, vy={teleop.vy:+.2f}, yaw={teleop.yaw_rate:+.2f}"
                        col2 = f"Base Z: {data.qpos[2]:.2f}m\nTilt: {tilt_deg:.1f}°\nTerminal: WASD, Space=STOP\nViewer: Arrow Keys / WASD"
                        viewer.set_texts([
                            (mujoco.mjtFontScale.mjFONTSCALE_100, mujoco.mjtGridPos.mjGRID_TOPLEFT, col1, col2)
                        ])
                    viewer.sync()

                # Update terminal HUD every 0.1s
                cur_wall = time.time()
                if cur_wall - last_hud_time > 0.10:
                    info = teleop.get_track_info(float(data.qpos[1]), float(data.qpos[0]))
                    print_hud(info, teleop, data, tilt_deg, data.time)
                    last_hud_time = cur_wall

                # Real-time synchronization
                expected_wall_time = (data.time - sim_start)
                actual_wall_time = time.time() - wall_start
                sleep_duration = expected_wall_time - actual_wall_time
                if sleep_duration > 0.0005:
                    time.sleep(sleep_duration)

    except KeyboardInterrupt:
        print("\n[INFO] Simulation stopped by user.")
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        teleop.stop_terminal_listener()
        if session:
            session.close(error)

    print("\n[INFO] Viewer closed.")


if __name__ == "__main__":
    main()
