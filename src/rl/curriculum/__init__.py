"""
课程学习模块

提供可复用的课程学习框架，支持多种强化学习任务的渐进式训练。

主要组件:
- BaseCurriculum: 课程学习基类
- ConfigurableCurriculum: 可配置的课程学习（支持单阶段和多阶段）
- WalkingCurriculum: 标准行走课程学习
"""

from .base import BaseCurriculum, CurriculumStage, ConfigurableCurriculum
from .walking import WalkingCurriculum

__all__ = [
    "BaseCurriculum",
    "CurriculumStage",
    "ConfigurableCurriculum",
    "WalkingCurriculum",
]
