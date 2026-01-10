from __future__ import annotations

from dataclasses import dataclass, field
import sys
from pathlib import Path

ISAACLAB_RL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ISAACLAB_RL_ROOT))

from jiyuan_tasks.utils.config_loader import ConfigDict
from jiyuan_tasks.utils.env_cfg_applier import apply_config_to_env_cfg


@dataclass
class DummyRewTerm:
    weight: float = 0.0
    params: dict = field(default_factory=dict)


@dataclass
class DummyRewardsCfg:
    track_lin_vel_xy: DummyRewTerm = field(default_factory=DummyRewTerm)
    alive: DummyRewTerm = field(default_factory=DummyRewTerm)
    feet_air_time: DummyRewTerm = field(default_factory=lambda: DummyRewTerm(params={"threshold": 0.5, "sensor_cfg": None}))
    undesired_contacts: DummyRewTerm = field(default_factory=lambda: DummyRewTerm(params={"threshold": 1.0, "sensor_cfg": None}))


@dataclass
class DummyDoneTerm:
    params: dict = field(default_factory=dict)


@dataclass
class DummyTerminationsCfg:
    fallen: DummyDoneTerm = field(default_factory=lambda: DummyDoneTerm(params={"min_height": 0.2, "max_roll": 1.0, "max_pitch": 1.0}))
    velocity_out_of_bounds: DummyDoneTerm = field(default_factory=lambda: DummyDoneTerm(params={"max_velocity": 10.0}))


@dataclass
class DummyEventsTerm:
    params: dict = field(default_factory=dict)
    interval_range_s: tuple[float, float] | None = None


@dataclass
class DummyEventsCfg:
    randomize_robot_mass: DummyEventsTerm = field(
        default_factory=lambda: DummyEventsTerm(params={"mass_distribution_params": (0.8, 1.2), "operation": "scale"})
    )
    push_robot: DummyEventsTerm = field(default_factory=lambda: DummyEventsTerm(params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}}))


@dataclass
class DummyRanges:
    lin_vel_x: tuple[float, float] = (-1.0, 1.0)
    lin_vel_y: tuple[float, float] = (-0.5, 0.5)
    ang_vel_z: tuple[float, float] = (-1.0, 1.0)


@dataclass
class DummyCommand:
    ranges: DummyRanges = field(default_factory=DummyRanges)
    resampling_time_range: tuple[float, float] = (10.0, 10.0)


@dataclass
class DummyCommands:
    base_velocity: DummyCommand = field(default_factory=DummyCommand)


@dataclass
class DummyEnvCfg:
    rewards: DummyRewardsCfg = field(default_factory=DummyRewardsCfg)
    terminations: DummyTerminationsCfg = field(default_factory=DummyTerminationsCfg)
    events: DummyEventsCfg = field(default_factory=DummyEventsCfg)
    commands: DummyCommands = field(default_factory=DummyCommands)


def test_apply_config_to_env_cfg_applies_reward_weights_and_thresholds():
    cfg = DummyEnvCfg()
    cfg.rewards.feet_air_time.params["sensor_cfg"] = type("SensorCfg", (), {"body_names": ".*"})()
    cfg.rewards.undesired_contacts.params["sensor_cfg"] = type("SensorCfg", (), {"body_names": ".*"})()
    train_cfg = ConfigDict(
        {
            "task": "velocity",
            "environment": {
                "velocity_tracking": {
                    "target_velocity_range": [-1.5, 1.5],
                    "target_lateral_velocity_range": [-0.8, 0.8],
                    "target_angular_velocity_range": [-2.0, 2.0],
                    "command_resample_time": 7.0,
                }
            },
            "rewards": {"velocity_tracking": {"track_lin_vel_xy": 1.5, "alive": 2.0}},
            "reward_params": {
                "velocity_tracking": {
                    "feet_air_time": {"threshold": 0.4, "sensor_cfg.body_names": ".*_foot_link"},
                    "undesired_contacts": {"threshold": 2.0, "sensor_cfg.body_names": "base_link"},
                }
            },
            "terminations": {"min_height": 0.15, "max_tilt": 0.7, "max_velocity": 30.0},
            "domain_randomization": {"randomize_mass": {"enable": False}},
        }
    )

    apply_config_to_env_cfg(cfg, train_cfg)

    assert cfg.rewards.track_lin_vel_xy.weight == 1.5
    assert cfg.rewards.alive.weight == 2.0
    assert cfg.rewards.feet_air_time.params["threshold"] == 0.4
    assert cfg.rewards.feet_air_time.params["sensor_cfg"].body_names == ".*_foot_link"
    assert cfg.rewards.undesired_contacts.params["threshold"] == 2.0
    assert cfg.rewards.undesired_contacts.params["sensor_cfg"].body_names == "base_link"
    assert cfg.terminations.fallen.params["min_height"] == 0.15
    assert cfg.terminations.fallen.params["max_roll"] == 0.7
    assert cfg.terminations.velocity_out_of_bounds.params["max_velocity"] == 30.0
    assert cfg.events.randomize_robot_mass is None
    assert cfg.commands.base_velocity.ranges.lin_vel_x == (-1.5, 1.5)
    assert cfg.commands.base_velocity.ranges.ang_vel_z == (-2.0, 2.0)
    assert cfg.commands.base_velocity.ranges.lin_vel_y == (-0.8, 0.8)
    assert cfg.commands.base_velocity.resampling_time_range == (7.0, 7.0)


def test_reward_params_task_section_overrides_section_key():
    cfg = DummyEnvCfg()
    cfg.rewards.feet_air_time.params["sensor_cfg"] = type("SensorCfg", (), {"body_names": ".*"})()
    cfg.rewards.undesired_contacts.params["sensor_cfg"] = type("SensorCfg", (), {"body_names": ".*"})()

    train_cfg = ConfigDict(
        {
            "task": "rough",
            "rewards": {"velocity_tracking": {"feet_air_time": 0.25}},
            "reward_params": {
                "velocity_tracking": {"feet_air_time": {"threshold": 0.5, "sensor_cfg.body_names": ".*_foot_link"}},
                "rough": {"feet_air_time": {"threshold": 0.4, "sensor_cfg.body_names": "right_foot_link"}},
            },
        }
    )

    apply_config_to_env_cfg(cfg, train_cfg)
    assert cfg.rewards.feet_air_time.weight == 0.25
    assert cfg.rewards.feet_air_time.params["threshold"] == 0.4
    assert cfg.rewards.feet_air_time.params["sensor_cfg"].body_names == "right_foot_link"
    # 叠加语义：未在 rough 中声明的项，沿用 velocity_tracking 默认
    assert cfg.rewards.undesired_contacts.params["threshold"] == 1.0


def test_reward_weights_task_section_overrides_section_key():
    cfg = DummyEnvCfg()
    train_cfg = ConfigDict(
        {
            "task": "rough",
            "rewards": {
                "velocity_tracking": {"track_lin_vel_xy": 1.0, "alive": 1.0},
                "rough": {"track_lin_vel_xy": 2.0},
            },
        }
    )

    apply_config_to_env_cfg(cfg, train_cfg)
    assert cfg.rewards.track_lin_vel_xy.weight == 2.0
    assert cfg.rewards.alive.weight == 1.0


def test_apply_standing_overrides_updates_height_params():
    @dataclass
    class DummyStandingRewardsCfg:
        height: DummyRewTerm = field(default_factory=DummyRewTerm)

    @dataclass
    class DummyStandingEnvCfg:
        rewards: DummyStandingRewardsCfg = field(default_factory=DummyStandingRewardsCfg)

        def __post_init__(self):
            self.rewards.height.params = {"target_height": 0.8, "tolerance": 0.05}

    cfg = DummyStandingEnvCfg()
    cfg.__post_init__()

    train_cfg = ConfigDict({"task": "standing", "environment": {"standing": {"target_height": 0.78, "height_tolerance": 0.06}}})
    apply_config_to_env_cfg(cfg, train_cfg)
    assert cfg.rewards.height.params["target_height"] == 0.78
    assert cfg.rewards.height.params["tolerance"] == 0.06
