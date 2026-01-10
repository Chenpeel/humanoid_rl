from __future__ import annotations

import sys
from pathlib import Path

ISAACLAB_RL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ISAACLAB_RL_ROOT))

from jiyuan_tasks.utils.config_loader import ConfigDict, apply_cli_overrides, merge_configs


def test_config_dict_dot_access():
    cfg = ConfigDict({"a": {"b": 1}})
    assert cfg.a.b == 1


def test_apply_cli_overrides_creates_paths():
    cfg = ConfigDict({})
    cfg = apply_cli_overrides(cfg, {"ppo.algorithm.learning_rate": 3e-4})
    assert cfg["ppo"]["algorithm"]["learning_rate"] == 3e-4


def test_merge_configs_recursive():
    base = ConfigDict({"a": {"b": 1, "c": 2}})
    override = ConfigDict({"a": {"c": 3}, "d": 4})
    merged = merge_configs(base, override)
    assert merged["a"]["b"] == 1
    assert merged["a"]["c"] == 3
    assert merged["d"] == 4
