"""
Managers 模块 —— 环境行为的可组合组件。

每个 Manager 负责环境运行时的某一方面：
  Rewards       → 奖励项计算与加权
  Observations  → 观测空间定义与数据提取
  Terminations  → 终止条件（跌倒、超时等）
  Commands      → 训练指令生成（目标速度、方向等）
"""

from .rewards import RewardsManager
from .observations import ObservationsManager
from .terminations import TerminationsManager
from .commands import CommandsManager
