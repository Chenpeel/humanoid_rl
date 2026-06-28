"""BaseEnvCfg —— 基础环境配置。

定义通用双足机器人环境的仿真参数：
- 仿真步长（dt, decimation）
- 物理参数（重力、地面）
- 机器人默认参数（PD 增益、关节限制）
"""

from __future__ import annotations

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


@configclass
class BaseEnvCfg(ManagerBasedRLEnvCfg):
    """双足机器人基础环境配置。

    所有任务配置（standing / walking / velocity_tracking）都继承此类。
    """

    # ---- 仿真参数 ----
    sim: SimulationCfg = SimulationCfg(
        dt=1.0 / 200.0,          # 200 Hz 物理仿真
        physx=PhysxCfg(
            gpu_max_rigid_contact_count=2**24,
            gpu_max_rigid_patch_count=2**20,
        ),
        render_interval=4,       # 每 4 步渲染一次（50 Hz 视觉）
    )

    # ---- 场景 ----
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=3.0)

    # ---- 动作空间 ----
    actions: ActionManager = ActionManager(num_actions=12, decimation=4)

    # ---- 事件（域随机化） ----
    events: EventManager = EventManager()

    # ---- 观测 ----
    observations: ObservationGroupManager = ObservationGroupManager()

    # ---- 奖励 ----
    rewards: RewardManager = RewardManager()

    # ---- 终止条件 ----
    terminations: TerminationManager = TerminationManager()

    # ---- 指令 ----
    commands: CommandManager = CommandManager()

    # ---- 课程学习 ----
    curriculum: CurriculumManager = CurriculumManager()

    def __post_init__(self):
        """子类在此覆盖具体配置。"""
        super().__post_init__()
