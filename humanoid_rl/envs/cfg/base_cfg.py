"""BaseEnvCfg —— 基础环境配置。

定义通用双足机器人环境的仿真参数。
仅在 Linux + GPU 环境下可用，macOS 上导入时优雅降级。
"""

from __future__ import annotations

_ISAAC_AVAILABLE = False
_isaac_classes = {}

try:
    from isaaclab.envs import ManagerBasedRLEnvCfg
    from isaaclab.managers import (
        ActionManager,
        EventManager,
        ObservationGroupManager,
        RewardManager,
        TerminationManager,
        CurriculumManager,
        CommandManager,
    )
    from isaaclab.scene import InteractiveSceneCfg
    from isaaclab.sim import PhysxCfg, SimulationCfg
    from isaaclab.utils import configclass

    _is_configclass = configclass
    _isaac_classes["ManagerBasedRLEnvCfg"] = ManagerBasedRLEnvCfg
    _isaac_classes["ActionManager"] = ActionManager
    _isaac_classes["EventManager"] = EventManager
    _isaac_classes["ObservationGroupManager"] = ObservationGroupManager
    _isaac_classes["RewardManager"] = RewardManager
    _isaac_classes["TerminationManager"] = TerminationManager
    _isaac_classes["CurriculumManager"] = CurriculumManager
    _isaac_classes["CommandManager"] = CommandManager
    _isaac_classes["InteractiveSceneCfg"] = InteractiveSceneCfg
    _isaac_classes["PhysxCfg"] = PhysxCfg
    _isaac_classes["SimulationCfg"] = SimulationCfg
    _ISAAC_AVAILABLE = True
except ImportError:
    def _is_configclass(cls):
        return cls

    class _Placeholder:
        """占位符 —— 模拟 Isaac Lab 配置基类，支持任意嵌套属性访问。"""
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
            self.__post_init__()

        def __post_init__(self):
            pass

        def __getattr__(self, name: str):
            # 自动创建嵌套占位符
            if name.startswith("_"):
                raise AttributeError(name)
            obj = _Placeholder()
            object.__setattr__(self, name, obj)
            return obj

        def __setattr__(self, name: str, value):
            object.__setattr__(self, name, value)

    _isaac_classes = {
        "ManagerBasedRLEnvCfg": _Placeholder,
        "ActionManager": _Placeholder,
        "EventManager": _Placeholder,
        "ObservationGroupManager": _Placeholder,
        "RewardManager": _Placeholder,
        "TerminationManager": _Placeholder,
        "CurriculumManager": _Placeholder,
        "CommandManager": _Placeholder,
        "InteractiveSceneCfg": _Placeholder,
        "PhysxCfg": _Placeholder,
        "SimulationCfg": _Placeholder,
    }


@_is_configclass
class BaseEnvCfg(_isaac_classes["ManagerBasedRLEnvCfg"]):
    """双足机器人基础环境配置。"""

    sim: object = _isaac_classes["SimulationCfg"](
        dt=1.0 / 200.0,
        physx=_isaac_classes["PhysxCfg"](
            gpu_max_rigid_contact_count=2**24,
            gpu_max_rigid_patch_count=2**20,
        ),
        render_interval=4,
    )
    scene: object = _isaac_classes["InteractiveSceneCfg"](num_envs=4096, env_spacing=3.0)
    actions: object = _isaac_classes["ActionManager"](num_actions=12, decimation=4)
    events: object = _isaac_classes["EventManager"]()
    observations: object = _isaac_classes["ObservationGroupManager"]()
    rewards: object = _isaac_classes["RewardManager"]()
    terminations: object = _isaac_classes["TerminationManager"]()
    commands: object = _isaac_classes["CommandManager"]()
    curriculum: object = _isaac_classes["CurriculumManager"]()

    def __post_init__(self):
        if _ISAAC_AVAILABLE:
            super().__post_init__()
