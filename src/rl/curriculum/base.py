"""
课程学习基类

定义课程学习框架的通用接口，可被各种任务（行走、跳跃、攀爬等）继承复用。

设计参考:
- Isaac Lab Curriculum: https://isaac-sim.github.io/IsaacLab/main/source/how-to/curriculums.html
- Gait-Conditioned RL: https://arxiv.org/abs/2505.20619
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional
from abc import ABC, abstractmethod


@dataclass
class CurriculumStage:
    """课程学习阶段配置

    Attributes:
        name: 阶段名称（用于日志显示）
        step_range: 步数范围 (start_step, end_step)
        reward_weights: 奖励权重字典
        env_config: 环境配置字典（速度范围、目标高度等）
        description: 阶段描述（可选）

    示例:
        >>> stage = CurriculumStage(
        ...     name="站立平衡",
        ...     step_range=(0, 50_000),
        ...     reward_weights={"alive": 5.0, "upright_bonus": 2.0},
        ...     env_config={"cmd_x_range": (0.0, 0.0), "target_height": 0.45},
        ...     description="学习保持直立不摔倒"
        ... )
    """
    name: str
    step_range: tuple  # (start_step, end_step)
    reward_weights: Dict[str, float]
    env_config: Dict[str, Any]
    description: Optional[str] = None


class BaseCurriculum(ABC):
    """课程学习基类

    定义课程学习框架的通用接口，子类需要实现 _define_stages 方法。

    属性:
        stages: 课程学习阶段列表
        current_stage: 当前阶段索引
        last_switch_step: 上次切换步数

    使用示例:
        >>> curriculum = WalkingCurriculum()
        >>> stage = curriculum.get_stage(30_000)
        >>> print(stage.name)
        站立平衡
        >>> curriculum.apply_to_env(env, 30_000)  # 应用阶段配置到环境
    """

    def __init__(self):
        self.stages = self._define_stages()
        self.current_stage = 0
        self.last_switch_step = 0
        self._validate_stages()

    @abstractmethod
    def _define_stages(self) -> list:
        """定义课程学习阶段（子类必须实现）

        Returns:
            CurriculumStage 列表，按训练顺序排列

        示例:
            >>> def _define_stages(self):
            ...     return [
            ...         CurriculumStage(
            ...             name="阶段1",
            ...             step_range=(0, 50_000),
            ...             reward_weights={...},
            ...             env_config={...}
            ...         ),
            ...         # ... 更多阶段
            ...     ]
        """
        pass

    def _validate_stages(self):
        """验证阶段配置的有效性

        检查:
        1. 至少有一个阶段
        2. 步数范围连续且不重叠
        3. 最后阶段的结束步数为 float('inf')
        """
        if not self.stages:
            raise ValueError("课程学习必须至少定义一个阶段")

        # 检查步数范围连续性
        for i in range(len(self.stages) - 1):
            current_end = self.stages[i].step_range[1]
            next_start = self.stages[i + 1].step_range[0]
            if current_end != next_start:
                raise ValueError(
                    f"阶段 {i} 和阶段 {i+1} 的步数范围不连续: "
                    f"{self.stages[i].step_range} vs {self.stages[i+1].step_range}"
                )

        # 检查最后阶段是否为无限步数
        if self.stages[-1].step_range[1] != float('inf'):
            raise ValueError(
                f"最后阶段的结束步数必须为 float('inf'), "
                f"当前为 {self.stages[-1].step_range[1]}"
            )

    def get_stage(self, current_step: int) -> CurriculumStage:
        """根据当前训练步数获取对应阶段

        Args:
            current_step: 当前训练步数

        Returns:
            对应的 CurriculumStage

        副作用:
            - 检测到阶段切换时打印日志
            - 更新 self.current_stage 和 self.last_switch_step
        """
        for i, stage in enumerate(self.stages):
            if stage.step_range[0] <= current_step < stage.step_range[1]:
                # 检测阶段切换
                if i != self.current_stage:
                    self._log_stage_switch(current_step, i)
                    self.current_stage = i
                    self.last_switch_step = current_step
                return stage

        # 超出范围，返回最后阶段
        return self.stages[-1]

    def _log_stage_switch(self, current_step: int, new_stage_idx: int):
        """记录阶段切换日志

        Args:
            current_step: 当前训练步数
            new_stage_idx: 新阶段索引
        """
        old_stage = self.stages[self.current_stage]
        new_stage = self.stages[new_stage_idx]

        print(f"\n{'='*70}")
        print(f"[课程学习] 阶段切换")
        print(f"  训练步数: {current_step:,}")
        print(f"  {old_stage.name} → {new_stage.name}")
        if new_stage.description:
            print(f"  目标: {new_stage.description}")
        print(f"{'='*70}\n")

    def apply_to_env(self, env, current_step: int):
        """将当前阶段配置应用到环境

        Args:
            env: 环境实例（需要有 reward_weights 和对应的环境参数属性）
            current_step: 当前训练步数

        副作用:
            - 更新 env.reward_weights
            - 更新 env 的环境配置参数（如 cmd_x_range、target_height 等）
        """
        stage = self.get_stage(current_step)

        # 更新奖励权重
        env.reward_weights = stage.reward_weights.copy()

        # 更新环境参数
        for key, value in stage.env_config.items():
            if hasattr(env, key):
                setattr(env, key, value)
            else:
                # 可选参数，不存在也不报错
                pass

    def get_stage_info(self, current_step: int) -> Dict[str, Any]:
        """获取当前阶段的详细信息（用于日志记录）

        Args:
            current_step: 当前训练步数

        Returns:
            包含阶段信息的字典

        示例:
            >>> info = curriculum.get_stage_info(30_000)
            >>> print(info)
            {
                'stage_name': '站立平衡',
                'stage_index': 0,
                'step_range': (0, 50000),
                'progress': 0.6,  # 当前阶段完成度
                'description': '学习保持直立不摔倒'
            }
        """
        stage = self.get_stage(current_step)
        start, end = stage.step_range

        # 计算阶段内进度（0-1）
        if end == float('inf'):
            progress = None  # 无限阶段无法计算进度
        else:
            progress = (current_step - start) / (end - start)

        return {
            'stage_name': stage.name,
            'stage_index': self.current_stage,
            'step_range': stage.step_range,
            'progress': progress,
            'description': stage.description,
        }
