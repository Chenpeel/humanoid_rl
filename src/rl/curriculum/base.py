"""
课程学习基类

定义课程学习框架的通用接口，以及基于配置文件的通用实现。

设计参考:
- Isaac Lab Curriculum: https://isaac-sim.github.io/IsaacLab/main/source/how-to/curriculums.html
- Gait-Conditioned RL: https://arxiv.org/abs/2505.20619
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union


# ============================================================================================
# ======================================= 数据结构 ============================================
# ============================================================================================

@dataclass
class CurriculumStage:
    """课程学习阶段配置

    Attributes:
        name: 阶段名称（用于日志显示）
        step_range: 步数范围 (start_step, end_step)
        reward_weights: 奖励权重字典
        env_config: 环境配置字典（速度范围、目标高度等）
        description: 阶段描述（可选）
    """

    name: str
    step_range: tuple  # (start_step, end_step)
    reward_weights: Dict[str, float]
    env_config: Dict[str, Any]
    description: Optional[str] = None

# ============================================================================================
# ===================================== END: 数据结构 ==========================================
# ============================================================================================


# ============================================================================================
# ======================================= 课程学习基类 =========================================
# ============================================================================================

class BaseCurriculum(ABC):
    """课程学习基类"""

    def __init__(self):
        self.stages = self._define_stages()
        self.current_stage = 0
        self.last_switch_step = 0
        self._validate_stages()

    @abstractmethod
    def _define_stages(self) -> List[CurriculumStage]:
        """定义课程学习阶段（子类必须实现）"""
        pass

    # --------------------------------------------------------------------------------------------

    def _validate_stages(self):
        """验证阶段配置的有效性"""
        if not self.stages:
            raise ValueError("课程学习必须至少定义一个阶段")

        for i in range(len(self.stages) - 1):
            current_end = self.stages[i].step_range[1]
            next_start = self.stages[i + 1].step_range[0]
            if current_end != next_start:
                raise ValueError(
                    f"阶段 {i} 和阶段 {i+1} 的步数范围不连续: "
                    f"{self.stages[i].step_range} vs {self.stages[i+1].step_range}"
                )

        if self.stages[-1].step_range[1] != float("inf"):
            raise ValueError(
                f"最后阶段的结束步数必须为 float('inf'), " f"当前为 {self.stages[-1].step_range[1]}"
            )

    # --------------------------------------------------------------------------------------------

    def get_stage(self, current_step: int) -> CurriculumStage:
        """根据当前训练步数获取对应阶段"""
        for i, stage in enumerate(self.stages):
            if stage.step_range[0] <= current_step < stage.step_range[1]:
                if i != self.current_stage:
                    self._log_stage_switch(current_step, i)
                    self.current_stage = i
                    self.last_switch_step = current_step
                return stage

        return self.stages[-1]

    # --------------------------------------------------------------------------------------------

    def _log_stage_switch(self, current_step: int, new_stage_idx: int):
        """记录阶段切换日志"""
        old_stage = self.stages[self.current_stage]
        new_stage = self.stages[new_stage_idx]

        print(f"\n{'='*70}")
        print(f"[课程学习] 阶段切换")
        print(f"  训练步数: {current_step:,}")
        print(f"  {old_stage.name} → {new_stage.name}")
        if new_stage.description:
            print(f"  目标: {new_stage.description}")
        print(f"{ '='*70}\n")

    # --------------------------------------------------------------------------------------------

    def apply_to_env(self, env, current_step: int):
        """将当前阶段配置应用到环境"""
        stage = self.get_stage(current_step)
        env.reward_weights = stage.reward_weights.copy()
        for key, value in stage.env_config.items():
            if hasattr(env, key):
                setattr(env, key, value)

    # --------------------------------------------------------------------------------------------

    def get_stage_info(self, current_step: int) -> Dict[str, Any]:
        """获取当前阶段的详细信息"""
        stage = self.get_stage(current_step)
        start, end = stage.step_range
        if end == float("inf"):
            progress = None
        else:
            progress = (current_step - start) / (end - start)

        return {
            "stage_name": stage.name,
            "stage_index": self.current_stage,
            "step_range": stage.step_range,
            "progress": progress,
            "description": stage.description,
        }

# ============================================================================================
# ===================================== END: 课程学习基类 ======================================
# ============================================================================================


# ============================================================================================
# ======================================= 可配置课程学习 ========================================
# ============================================================================================

class ConfigurableCurriculum(BaseCurriculum):
    """可配置的课程学习通用实现

    支持两种配置模式：
    1. 简单模式：仅提供权重字典，视为单阶段无限时长的训练。
    2. 高级模式：提供完整的阶段列表定义。
    """

    def __init__(
        self,
        config: Union[Dict[str, float], Dict[str, Any]],
        env_config: Dict[str, Any] = None,
        default_stage_name: str = "ConfigurableStage",
    ):
        self._raw_config = config
        self._global_env_config = env_config or {}
        self._default_stage_name = default_stage_name
        super().__init__()

    # --------------------------------------------------------------------------------------------

    def _define_stages(self) -> List[CurriculumStage]:
        # 模式 1: 高级模式（多阶段）
        if "stages" in self._raw_config and isinstance(self._raw_config["stages"], list):
            stages = []
            raw_stages = self._raw_config["stages"]
            
            for i, stage_def in enumerate(raw_stages):
                start_step = stage_def.get("start_step", 0)
                # 自动推断 start_step
                if i > 0 and "start_step" not in stage_def:
                    start_step = stages[-1].step_range[1]
                
                end_step = stage_def.get("end_step", float("inf"))
                if isinstance(end_step, str) and end_step.lower() == "inf":
                    end_step = float("inf")
                
                # 合并环境配置 (阶段配置 > 全局配置)
                current_env_config = self._global_env_config.copy()
                current_env_config.update(stage_def.get("env_config", {}))

                stages.append(
                    CurriculumStage(
                        name=stage_def.get("name", f"Stage_{i+1}"),
                        step_range=(start_step, end_step),
                        reward_weights=stage_def["reward_weights"],
                        env_config=current_env_config,
                        description=stage_def.get("description", ""),
                    )
                )
            return stages

        # 模式 2: 简单模式（单阶段 - 仅权重）
        else:
            return [
                CurriculumStage(
                    name=self._default_stage_name,
                    step_range=(0, float("inf")),
                    reward_weights=self._raw_config,
                    env_config=self._global_env_config,
                    description="从简单配置文件加载的单阶段训练",
                ),
            ]

# ============================================================================================
# ===================================== END: 可配置课程学习 =====================================
# ============================================================================================
