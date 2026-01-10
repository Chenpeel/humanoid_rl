"""
Jiyuan 机器人任务定义模块

本包包含 Jiyuan 双足机器人的 Isaac Lab 任务环境配置与管理器函数。

默认行为:
- 当 `gymnasium` 可用时，导入 `jiyuan_tasks` 会自动将环境注册到 Gym registry（与训练脚本兼容）。

纯 Python 工具兼容:
- 允许在没有 `gymnasium` / Isaac Sim 运行时依赖的环境里导入 `jiyuan_tasks.utils.*`（例如跑单测/做静态分析）。
"""

from __future__ import annotations

from typing import Optional

try:
    import gymnasium as gym
except ModuleNotFoundError:  # pragma: no cover
    gym = None  # type: ignore[assignment]


def register_envs(_gym: Optional[object] = None) -> None:
    """注册所有 Jiyuan 环境到 Gym registry。"""
    g = _gym or gym
    if g is None:  # pragma: no cover
        raise ModuleNotFoundError("未找到 gymnasium，无法注册环境。请在 Isaac Lab Python 环境中运行。")

    # 速度跟踪环境（主要训练任务）
    g.register(
        id="Isaac-Jiyuan-Velocity-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        kwargs={"env_cfg_entry_point": "jiyuan_tasks.envs.cfg:VELOCITY_TRACKING_ENV_CFG"},
        disable_env_checker=True,
    )

    # 站立平衡环境（预训练/调试）
    g.register(
        id="Isaac-Jiyuan-Standing-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        kwargs={"env_cfg_entry_point": "jiyuan_tasks.envs.cfg:STANDING_ENV_CFG"},
        disable_env_checker=True,
    )

    # 粗糙地形环境（复杂地形训练）
    g.register(
        id="Isaac-Jiyuan-Rough-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        kwargs={"env_cfg_entry_point": "jiyuan_tasks.envs.cfg:JIYUAN_ROUGH_ENV_CFG"},
        disable_env_checker=True,
    )

    # 平坦地形环境（简化训练）
    g.register(
        id="Isaac-Jiyuan-Flat-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        kwargs={"env_cfg_entry_point": "jiyuan_tasks.envs.cfg:JIYUAN_FLAT_ENV_CFG"},
        disable_env_checker=True,
    )

    # 分阶段课程学习环境
    g.register(
        id="Isaac-Jiyuan-Curriculum-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        kwargs={"env_cfg_entry_point": "jiyuan_tasks.envs.cfg:CURRICULUM_ENV_CFG"},
        disable_env_checker=True,
    )


if gym is not None:
    register_envs(gym)


__all__ = [
    "register_envs",
]
