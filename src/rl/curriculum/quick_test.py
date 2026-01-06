"""
快速测试的课程学习实现

调整阶段切换点以适应小批量快速测试：
1. 站立平衡（0-500k steps）：学习保持直立不摔倒
2. 低速行走（500k-1M steps）：学习基本步态和低速前进
3. 全速行走（1M+ steps）：跟踪任意速度命令并优化性能

设计原则：
- 每个阶段至少 100+ 梯度更新（确保充分学习）
- 基于 batch_size=16384 (256 envs × 64 steps) 设计
- Stage 1: 30 updates × 8 grad = 240 梯度更新（学会站立）
- Stage 2: 30 updates × 8 grad = 240 梯度更新（学会行走）
- Stage 3: 60+ updates × 8 grad = 480+ 梯度更新（优化性能）
"""

from .base import BaseCurriculum, CurriculumStage


class QuickTestCurriculum(BaseCurriculum):
    """快速测试的三阶段课程学习

    专为快速验证设计，阶段切换点基于小批量训练调整。

    使用示例:
        >>> from rl.curriculum.quick_test import QuickTestCurriculum
        >>> curriculum = QuickTestCurriculum()
        >>> # 在训练循环中
        >>> for step in range(total_steps):
        ...     curriculum.apply_to_env(env, step)
        ...     # ... 正常训练步骤 ...
    """

    def _define_stages(self) -> list:
        """定义快速测试的三阶段课程学习配置"""
        # 从 rewards 模块导入权重配置
        from ..rewards.walking_rewards import (
            STAGE1_WEIGHTS,
            STAGE2_WEIGHTS,
            STAGE3_WEIGHTS,
        )

        return [
            # ==================== 阶段 1: 站立平衡（0-500k steps）====================
            CurriculumStage(
                name="站立平衡",
                step_range=(0, 500_000),  # 调整：原 50k → 500k
                reward_weights=STAGE1_WEIGHTS,
                env_config={
                    # 零速度命令（站立不动）
                    "cmd_x_range": (0.0, 0.0),
                    "cmd_y_range": (0.0, 0.0),
                    "cmd_yaw_range": (0.0, 0.0),
                    # 目标高度稍高（匹配初始化高度）
                    "target_height": 0.45,
                },
                description="学习保持直立姿态、抵抗扰动、建立基本平衡控制能力",
            ),
            # ==================== 阶段 2: 低速行走（500k-1M steps）====================
            CurriculumStage(
                name="低速行走",
                step_range=(500_000, 1_000_000),  # 调整：原 50k-150k → 500k-1M
                reward_weights=STAGE2_WEIGHTS,
                env_config={
                    # 低速命令范围
                    "cmd_x_range": (0.1, 0.3),  # 前向 0.1-0.3 m/s
                    "cmd_y_range": (-0.1, 0.1),  # 侧向 ±0.1 m/s
                    "cmd_yaw_range": (-0.3, 0.3),  # 转向 ±0.3 rad/s
                    # 目标高度略低（更真实的行走高度）
                    "target_height": 0.40,
                },
                description="学习双脚交替接触、建立基本步态模式、实现低速稳定前进",
            ),
            # ==================== 阶段 3: 全速行走（1M+ steps）====================
            CurriculumStage(
                name="全速行走",
                step_range=(1_000_000, float("inf")),  # 调整：原 150k+ → 1M+
                reward_weights=STAGE3_WEIGHTS,
                env_config={
                    # 全速度范围（包括后退）
                    "cmd_x_range": (-0.2, 0.8),  # 前向 -0.2~0.8 m/s（含后退）
                    "cmd_y_range": (-0.3, 0.3),  # 侧向 ±0.3 m/s
                    "cmd_yaw_range": (-1.0, 1.0),  # 转向 ±1.0 rad/s
                    # 恢复标准目标高度
                    "target_height": 0.35,
                },
                description="跟踪任意速度命令、优化步态质量和能量效率、提高鲁棒性",
            ),
        ]
