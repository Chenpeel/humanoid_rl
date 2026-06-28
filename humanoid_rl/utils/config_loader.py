"""config_loader —— YAML 配置加载与合并。

支持 Hydra / OmegaConf 风格的配置管理，按 robot + task 两层合并。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_robot_config(robot_name: str) -> dict[str, Any]:
    """加载机器人配置。

    Args:
        robot_name: 机器人名称（如 h1, h2）

    Returns:
        机器人配置字典
    """
    config_path = Path(__file__).parent.parent.parent / f"configs/robots/{robot_name}.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Robot config not found: {config_path}")

    with open(config_path) as f:
        return yaml.safe_load(f)


def load_task_config(task_name: str) -> dict[str, Any]:
    """加载任务配置（standing / walking / velocity_tracking 的默认参数）。

    Args:
        task_name: 任务名称

    Returns:
        任务配置字典
    """
    config_path = (
        Path(__file__).parent.parent.parent / f"configs/tasks/{task_name}.yaml"
    )
    if config_path.exists():
        with open(config_path) as f:
            return yaml.safe_load(f)
    # 任务无独立文件时，返回空（使用代码默认值）
    return {}


def merge_configs(base: dict, override: dict) -> dict:
    """深度合并两个配置字典。override 覆盖 base。"""
    merged = base.copy()
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
    return merged
