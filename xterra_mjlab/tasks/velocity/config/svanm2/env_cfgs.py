"""xTerra SvanM2 velocity environment configurations.

45-dim actor observation, base_lin_vel-last critic layout. Foot geoms/sites are
``*_foot``; leg bodies are ``*_thigh_link`` / ``*_shank_link``. Illegal-contact
(rough base only) uses body-mode contact sensors. This package registers only
the flat variant.
"""

import math

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers import TerminationTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import (
    ContactMatch,
    ContactSensorCfg,
    ObjRef,
    RayCastSensorCfg,
    RingPatternCfg,
    TerrainHeightSensorCfg,
)
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from xterra_mjlab.assets.svanm2 import get_svanm2_robot_cfg, SVANM2_ACTION_SCALE


def svanm2_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    cfg = make_velocity_env_cfg()

    cfg.sim.mujoco.ccd_iterations = 500
    cfg.sim.mujoco.impratio = 10
    cfg.sim.mujoco.cone = "elliptic"
    cfg.sim.contact_sensor_maxmatch = 500

    cfg.scene.entities = {"robot": get_svanm2_robot_cfg(delayed=not play)}

    for sensor in cfg.scene.sensors or ():
        if sensor.name == "terrain_scan":
            assert isinstance(sensor, RayCastSensorCfg)
            assert isinstance(sensor.frame, ObjRef)
            sensor.frame.name = "base"

    foot_names = ("FL_foot", "FR_foot", "RL_foot", "RR_foot")
    site_names = foot_names
    geom_names = foot_names

    for sensor in cfg.scene.sensors or ():
        if sensor.name == "foot_height_scan":
            assert isinstance(sensor, TerrainHeightSensorCfg)
            sensor.frame = tuple(
                ObjRef(type="site", name=s, entity="robot") for s in site_names
            )
            sensor.pattern = RingPatternCfg.single_ring(radius=0.04, num_samples=4)

    feet_ground_cfg = ContactSensorCfg(
        name="feet_ground_contact",
        primary=ContactMatch(mode="geom", pattern=geom_names, entity="robot"),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
        track_air_time=True,
    )
    self_collision_cfg = ContactSensorCfg(
        name="self_collision",
        primary=ContactMatch(mode="subtree", pattern="base", entity="robot"),
        secondary=ContactMatch(mode="subtree", pattern="base", entity="robot"),
        fields=("found", "force"),
        reduce="none",
        num_slots=1,
        history_length=4,
    )
    thigh_ground_cfg = ContactSensorCfg(
        name="thigh_ground_touch",
        primary=ContactMatch(mode="body", entity="robot", pattern="(FL|FR|RL|RR)_thigh_link"),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="none",
        num_slots=1,
        history_length=4,
    )
    shank_ground_cfg = ContactSensorCfg(
        name="shank_ground_touch",
        primary=ContactMatch(mode="body", entity="robot", pattern="(FL|FR|RL|RR)_shank_link"),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="none",
        num_slots=1,
        history_length=4,
    )
    trunk_ground_cfg = ContactSensorCfg(
        name="trunk_ground_touch",
        primary=ContactMatch(mode="body", entity="robot", pattern="base"),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="none",
        num_slots=1,
        history_length=4,
    )
    cfg.scene.sensors = (cfg.scene.sensors or ()) + (
        feet_ground_cfg,
        self_collision_cfg,
        thigh_ground_cfg,
        shank_ground_cfg,
        trunk_ground_cfg,
    )

    if (
        cfg.scene.terrain is not None
        and cfg.scene.terrain.terrain_generator is not None
    ):
        cfg.scene.terrain.terrain_generator.curriculum = True

    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg)
    joint_pos_action.scale = SVANM2_ACTION_SCALE
    joint_pos_action.clip={
        r".*_hip_joint":(-0.735,0.735),
        r".*F[LR]_thigh_joint":(-1.5708,3.4907),
        r".*R[LR]_thigh_joint":(-0.5236,4.5379),
        r".*_calf_joint":(-2.7227,-0.88776),
    }
    cfg.viewer.body_name = "base"
    cfg.viewer.distance = 2.0
    cfg.viewer.elevation = -10.0

    # Actor observation layout (45-dim per frame, 6-frame history).
    cfg.observations["actor"] = ObservationGroupCfg(
        terms={
            "command": ObservationTermCfg(
                func=mdp.generated_commands,
                params={"command_name": "twist"},
            ),
            "base_ang_vel": ObservationTermCfg(
                func=mdp.builtin_sensor,
                params={"sensor_name": "robot/imu_ang_vel"},
                scale=0.25,
                noise=Unoise(n_min=-0.2, n_max=0.2),
            ),
            "projected_gravity": ObservationTermCfg(
                func=mdp.projected_gravity,
                noise=Unoise(n_min=-0.05, n_max=0.05),
            ),
            "joint_pos": ObservationTermCfg(
                func=mdp.joint_pos_rel,
                noise=Unoise(n_min=-0.01, n_max=0.01),
            ),
            "joint_vel": ObservationTermCfg(
                func=mdp.joint_vel_rel,
                scale=0.05,
                noise=Unoise(n_min=-1.5, n_max=1.5),
            ),
            "actions": ObservationTermCfg(func=mdp.last_action),
        },
        concatenate_terms=True,
        enable_corruption=True,
        history_length=6,
    )

    cfg.observations["critic"] = ObservationGroupCfg(
        terms={
            "command": ObservationTermCfg(
                func=mdp.generated_commands,
                params={"command_name": "twist"},
            ),
            "base_ang_vel": ObservationTermCfg(
                func=mdp.builtin_sensor,
                params={"sensor_name": "robot/imu_ang_vel"},
                scale=0.25,
            ),
            "projected_gravity": ObservationTermCfg(func=mdp.projected_gravity),
            "joint_pos": ObservationTermCfg(func=mdp.joint_pos_rel),
            "joint_vel": ObservationTermCfg(func=mdp.joint_vel_rel, scale=0.05),
            "actions": ObservationTermCfg(func=mdp.last_action),
            "base_lin_vel": ObservationTermCfg(
                func=mdp.builtin_sensor,
                params={"sensor_name": "robot/imu_lin_vel"},
            ),
            "height_scan": ObservationTermCfg(
                func=envs_mdp.height_scan,
                params={"sensor_name": "terrain_scan"},
            ),
            "foot_air_time": ObservationTermCfg(
                func=mdp.foot_air_time,
                params={"sensor_name": "feet_ground_contact"},
            ),
            "foot_contact": ObservationTermCfg(
                func=mdp.foot_contact,
                params={"sensor_name": "feet_ground_contact"},
            ),
            "foot_contact_forces": ObservationTermCfg(
                func=mdp.foot_contact_forces,
                params={"sensor_name": "feet_ground_contact"},
            ),
        },
        concatenate_terms=True,
        enable_corruption=False,
    )

    twist_cmd = cfg.commands["twist"]
    assert isinstance(twist_cmd, UniformVelocityCommandCfg)
    twist_cmd.ranges.lin_vel_x = (-1.0, 1.0)
    twist_cmd.ranges.lin_vel_y = (-1.0, 1.0)
    twist_cmd.ranges.ang_vel_z = (-2.0, 2.0)
    twist_cmd.resampling_time_range = (10.0, 10.0)
    twist_cmd.rel_standing_envs = 0.1
    twist_cmd.rel_heading_envs = 0.5
    twist_cmd.heading_control_stiffness = 0.5

    cfg.events["foot_friction"].params["asset_cfg"].geom_names = geom_names
    cfg.events["foot_friction"].params["ranges"] = (0.3, 1.5)
    cfg.events["foot_friction"].params["operation"] = "abs"
    cfg.events["base_com"].params["asset_cfg"].body_names = ("base",)

    cfg.events["base_mass"] = EventTermCfg(
        func=mdp.dr.body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=("base",)),
            "ranges": (-1.0, 3.0),
            "operation": "add",
        },
    )
    cfg.events["pd_gains"] = EventTermCfg(
        func=mdp.dr.pd_gains,
        mode="startup",
        params={"kp_range": (0.8, 1.2), "kd_range": (0.8, 1.2), "operation": "scale"},
    )

    cfg.rewards["track_linear_velocity"].weight = 6.0
    cfg.rewards["track_linear_velocity"].params["std"] = math.sqrt(0.25)
    cfg.rewards["track_angular_velocity"].weight = 3.0
    cfg.rewards["track_angular_velocity"].params["std"] = math.sqrt(0.5)
    cfg.rewards["action_rate_l2"].weight = -0.15

    cfg.rewards["upright"].weight = 0.5
    cfg.rewards["upright"].params["asset_cfg"].body_names = ("base",)
    cfg.rewards["upright"].params["terrain_sensor_names"] = ("terrain_scan",)

    cfg.rewards["pose"].weight = 0.25
    cfg.rewards["pose"].params["std_standing"] = {
        r".*(FR|FL|RR|RL)_(hip|thigh)_joint.*": 0.05,
        r".*(FR|FL|RR|RL)_calf_joint.*": 0.1,
    }
    cfg.rewards["pose"].params["std_walking"] = {
        r".*(FR|FL|RR|RL)_(hip|thigh)_joint.*": 0.3,
        r".*(FR|FL|RR|RL)_calf_joint.*": 0.6,
    }
    cfg.rewards["pose"].params["std_running"] = {
        r".*(FR|FL|RR|RL)_(hip|thigh)_joint.*": 0.3,
        r".*(FR|FL|RR|RL)_calf_joint.*": 0.6,
    }

    cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("base",)
    cfg.rewards["body_ang_vel"].weight = 0.0
    cfg.rewards["angular_momentum"].weight = 0.0
    cfg.rewards["air_time"].weight = 0.0

    for reward_name in ["foot_clearance", "foot_slip"]:
        cfg.rewards[reward_name].params["asset_cfg"].site_names = site_names

    cfg.rewards["self_collisions"] = RewardTermCfg(
        func=mdp.self_collision_cost,
        weight=-0.1,
        params={"sensor_name": self_collision_cfg.name},
    )
    cfg.rewards["shank_collision"] = RewardTermCfg(
        func=mdp.self_collision_cost,
        weight=-0.1,
        params={"sensor_name": shank_ground_cfg.name},
    )
    cfg.rewards["trunk_collision"] = RewardTermCfg(
        func=mdp.self_collision_cost,
        weight=-0.1,
        params={"sensor_name": trunk_ground_cfg.name},
    )

    cfg.terminations.pop("fell_over", None)
    cfg.terminations["illegal_contact"] = TerminationTermCfg(
        func=mdp.illegal_contact,
        params={"sensor_name": thigh_ground_cfg.name},
    )

    if play:
        cfg.episode_length_s = int(1e9)
        cfg.observations["actor"].enable_corruption = False
        cfg.events.pop("push_robot", None)
        cfg.terminations.pop("out_of_terrain_bounds", None)
        cfg.curriculum = {}
        cfg.events["randomize_terrain"] = EventTermCfg(
            func=envs_mdp.randomize_terrain,
            mode="reset",
            params={},
        )
        if cfg.scene.terrain is not None:
            if cfg.scene.terrain.terrain_generator is not None:
                cfg.scene.terrain.terrain_generator.curriculum = False
                cfg.scene.terrain.terrain_generator.num_cols = 5
                cfg.scene.terrain.terrain_generator.num_rows = 5
                cfg.scene.terrain.terrain_generator.border_width = 10.0

    return cfg


def svanm2_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    cfg = svanm2_rough_env_cfg(play=play)

    cfg.sim.njmax = 300
    cfg.sim.mujoco.ccd_iterations = 50
    cfg.sim.contact_sensor_maxmatch = 64
    cfg.sim.nconmax = None

    assert cfg.scene.terrain is not None
    cfg.scene.terrain.terrain_type = "plane"
    cfg.scene.terrain.terrain_generator = None

    remove_sensors = {
        "terrain_scan",
        "foot_height_scan",
        "self_collision",
        "thigh_ground_touch",
        "shank_ground_touch",
        "trunk_ground_touch",
    }
    cfg.scene.sensors = tuple(
        s for s in (cfg.scene.sensors or ()) if s.name not in remove_sensors
    )
    cfg.observations["critic"].terms.pop("height_scan", None)
    cfg.rewards["upright"].params.pop("terrain_sensor_names", None)

    for key in (
        "self_collisions",
        "shank_collision",
        "trunk_collision",
        "foot_clearance",
        "foot_swing_height",
    ):
        cfg.rewards.pop(key, None)

    cfg.terminations.pop("illegal_contact", None)
    cfg.terminations.pop("out_of_terrain_bounds", None)
    cfg.terminations["fell_over"] = TerminationTermCfg(
        func=mdp.bad_orientation,
        params={"limit_angle": math.radians(70.0)},
    )

    cfg.curriculum.pop("terrain_levels", None)

    if play:
        twist_cmd = cfg.commands["twist"]
        assert isinstance(twist_cmd, UniformVelocityCommandCfg)
        twist_cmd.ranges.lin_vel_x = (-1.5, 2.0)
        twist_cmd.ranges.ang_vel_z = (-0.7, 0.7)

    return cfg
