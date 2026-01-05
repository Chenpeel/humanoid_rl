"""
Jiyuan 行走任务的课程学习实现

三阶段渐进式训练:
1. 站立平衡（0-50k steps）：学习保持直立不摔倒
2. 低速行走（50k-150k steps）：学习基本步态和低速前进
3. 全速行走（150k+ steps）：跟踪任意速度命令并优化性能

设计参考:
- Isaac Lab H1 Rough: 使用类似的多阶段课程学习
- Gait-Conditioned RL: 渐进式增加难度和任务复杂度
"""

from .base import BaseCurriculum, CurriculumStage


class WalkingCurriculum(BaseCurriculum):
    """Jiyuan 行走任务的三阶段课程学习

    使用示例:
        >>> curriculum = WalkingCurriculum()
        >>> # 在训练循环中
        >>> for step in range(total_steps):
        ...     curriculum.apply_to_env(env, step)
        ...     # ... 正常训练步骤 ...
    """

    def _define_stages(self) -> list:
        """定义三阶段课程学习配置"""
        # 从 rewards 模块导入权重配置
        from ..rewards.walking_rewards import (
            STAGE1_WEIGHTS,
            STAGE2_WEIGHTS,
            STAGE3_WEIGHTS
        )

        return [
            # ==================== 阶段 1: 站立平衡（0-50k steps）====================
            CurriculumStage(
                name="站立平衡",
                step_range=(0, 50_000),
                reward_weights=STAGE1_WEIGHTS,
                env_config={
                    # 零速度命令（站立不动）
                    "cmd_x_range": (0.0, 0.0),
                    "cmd_y_range": (0.0, 0.0),
                    "cmd_yaw_range": (0.0, 0.0),
                    # 目标高度稍高（匹配初始化高度）
                    "target_height": 0.45,
                },
                description="学习保持直立姿态、抵抗扰动、建立基本平衡控制能力"
            ),

            # ==================== 阶段 2: 低速行走（50k-150k steps）====================
            CurriculumStage(
                name="低速行走",
                step_range=(50_000, 150_000),
                reward_weights=STAGE2_WEIGHTS,
                env_config={
                    # 低速命令范围
                    "cmd_x_range": (0.1, 0.3),    # 前向 0.1-0.3 m/s
                    "cmd_y_range": (-0.1, 0.1),   # 侧向 ±0.1 m/s
                    "cmd_yaw_range": (-0.3, 0.3), # 转向 ±0.3 rad/s
                    # 目标高度略低（更真实的行走高度）
                    "target_height": 0.40,
                },
                description="学习双脚交替接触、建立基本步态模式、实现低速稳定前进"
            ),

            # ==================== 阶段 3: 全速行走（150k+ steps）====================
            CurriculumStage(
                name="全速行走",
                step_range=(150_000, float('inf')),
                reward_weights=STAGE3_WEIGHTS,
                env_config={
                    # 全速度范围（包括后退）
                    "cmd_x_range": (-0.2, 0.8),   # 前向 -0.2~0.8 m/s（含后退）
                    "cmd_y_range": (-0.3, 0.3),   # 侧向 ±0.3 m/s
                    "cmd_yaw_range": (-1.0, 1.0), # 转向 ±1.0 rad/s
                    # 恢复标准目标高度
                    "target_height": 0.35,
                },
                description="跟踪任意速度命令、优化步态质量和能量效率、提高鲁棒性"
            ),
        ]
