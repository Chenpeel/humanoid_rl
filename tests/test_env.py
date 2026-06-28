"""Tests for humanoid_rl."""

import pytest


def test_import():
    """验证包可导入。"""
    import humanoid_rl
    assert humanoid_rl.__version__ == "0.1.0"


def test_config_loader():
    """验证配置加载器。"""
    from humanoid_rl.utils.config_loader import merge_configs

    base = {"a": 1, "b": {"c": 2}}
    override = {"b": {"c": 3}, "d": 4}
    merged = merge_configs(base, override)
    assert merged["a"] == 1
    assert merged["b"]["c"] == 3
    assert merged["d"] == 4


def test_env_cfg_import():
    """验证环境配置可导入。"""
    from humanoid_rl.envs.cfg.standing_cfg import StandingEnvCfg
    from humanoid_rl.envs.cfg.walking_cfg import WalkingEnvCfg
    from humanoid_rl.envs.cfg.velocity_cfg import VelocityTrackingEnvCfg

    # 实例化不应报错
    standing = StandingEnvCfg()
    assert standing.task_name == "standing"

    walking = WalkingEnvCfg()
    assert walking.task_name == "walking"

    vel = VelocityTrackingEnvCfg()
    assert vel.task_name == "velocity_tracking"


def test_managers_import():
    """验证 Managers 可导入。"""
    from humanoid_rl.managers.rewards import RewardsManager
    from humanoid_rl.managers.observations import ObservationsManager
    from humanoid_rl.managers.terminations import TerminationsManager
    from humanoid_rl.managers.commands import CommandsManager
    assert True  # import 成功


def test_ppo_cfg():
    """验证 PPO 配置默认值。"""
    from humanoid_rl.agents.ppo_cfg import PPOCfg
    cfg = PPOCfg()
    assert cfg.num_envs == 4096
    assert cfg.learning_rate == 1e-3
    assert cfg.num_steps_per_env == 24
