from __future__ import annotations

import sys
from pathlib import Path

import pytest

ISAACLAB_RL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ISAACLAB_RL_ROOT))

from jiyuan_tasks.utils.config_loader import ConfigDict
from jiyuan_tasks.utils.ros_bridge import build_ankle_indices, prepare_servo_command_entries


def test_build_ankle_indices_from_config_dict():
    cfg = ConfigDict(
        {
            "action_mapping": {
                "ankle_indices": {
                    "left": {"roll": 9, "pitch": 10, "yaw": 11},
                    "right": {"roll": 12, "pitch": 13, "yaw": 14},
                }
            }
        }
    )

    indices = build_ankle_indices(cfg)
    assert indices == {"left": [9, 10, 11], "right": [12, 13, 14]}


def test_build_ankle_indices_requires_mapping_shape():
    cfg = ConfigDict({"action_mapping": {"ankle_indices": []}})
    with pytest.raises(ValueError):
        build_ankle_indices(cfg)


def test_prepare_servo_command_entries_applies_default_speed():
    entries = prepare_servo_command_entries(
        commands=[{"id": 9, "position": 1500}, {"servo_id": 10, "position": 1600, "speed": 80}],
        default_speed=100,
    )

    assert entries[0].servo_id == 9
    assert entries[0].position == 1500
    assert entries[0].speed == 100
    assert entries[1].servo_id == 10
    assert entries[1].speed == 80


def test_prepare_servo_command_entries_speed_override_and_clamp():
    entries = prepare_servo_command_entries(
        commands=[{"id": -1, "position": 70000, "speed": -30}],
        default_speed=100,
        speed_override=120,
    )

    assert entries[0].servo_id == 0
    assert entries[0].position == 65535
    assert entries[0].speed == 120


def test_prepare_servo_command_entries_requires_position():
    with pytest.raises(ValueError):
        prepare_servo_command_entries(commands=[{"id": 9}], default_speed=100)
