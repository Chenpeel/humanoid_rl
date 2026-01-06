"""
课程学习模块

提供可复用的课程学习框架，支持多种强化学习任务的渐进式训练。

主要组件:
- BaseCurriculum: 课程学习基类
- WalkingCurriculum: 行走任务的三阶段课程学习（标准）
- QuickTestCurriculum: 快速测试的三阶段课程学习（延长 Stage 1）
"""

from .base import BaseCurriculum, CurriculumStage
from .quick_test import QuickTestCurriculum
from .walking import WalkingCurriculum

__all__ = [
    "BaseCurriculum",
    "CurriculumStage",
    "WalkingCurriculum",
    "QuickTestCurriculum",
]
