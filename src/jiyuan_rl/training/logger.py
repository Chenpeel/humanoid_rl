"""
日志系统：TensorBoard + Rich终端展示
"""

import jax.numpy as jp
from pathlib import Path
from typing import Dict, Any, Optional
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn, TimeElapsedColumn
from rich import box
import time

try:
    from tensorboardX import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError:
    try:
        from torch.utils.tensorboard import SummaryWriter
        HAS_TENSORBOARD = True
    except ImportError:
        HAS_TENSORBOARD = False
        print("警告: 未安装tensorboardX或torch，TensorBoard日志将被禁用")


class Logger:
    """训练日志记录器"""

    def __init__(
        self,
        log_dir: str,
        use_tensorboard: bool = True,
        use_rich: bool = True,
    ):
        """初始化日志记录器

        Args:
            log_dir: 日志目录
            use_tensorboard: 是否使用TensorBoard
            use_rich: 是否使用Rich终端输出
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.use_tensorboard = use_tensorboard and HAS_TENSORBOARD
        self.use_rich = use_rich

        # TensorBoard writer
        if self.use_tensorboard:
            self.writer = SummaryWriter(str(self.log_dir))
        else:
            self.writer = None

        # Rich console
        if self.use_rich:
            self.console = Console()
        else:
            self.console = None

        # 记录开始时间
        self.start_time = time.time()
        self.last_log_time = self.start_time

    def log_scalars(
        self,
        metrics: Dict[str, float],
        step: int,
        prefix: str = "",
    ):
        """记录标量指标

        Args:
            metrics: 指标字典
            step: 当前步数
            prefix: 指标名前缀
        """
        if self.use_tensorboard and self.writer:
            for key, value in metrics.items():
                full_key = f"{prefix}/{key}" if prefix else key
                self.writer.add_scalar(full_key, value, step)

    def log_histogram(
        self,
        name: str,
        values: jp.ndarray,
        step: int,
    ):
        """记录直方图

        Args:
            name: 直方图名称
            values: 数值数组
            step: 当前步数
        """
        if self.use_tensorboard and self.writer:
            import numpy as np
            self.writer.add_histogram(name, np.array(values), step)

    def print_training_status(
        self,
        step: int,
        total_steps: int,
        env_steps: int,
        metrics: Dict[str, Any],
    ):
        """打印训练状态（使用Rich）

        Args:
            step: 当前训练步数
            total_steps: 总训练步数
            env_steps: 环境交互总步数
            metrics: 当前指标
        """
        if not self.use_rich or not self.console:
            return

        # 计算耗时
        current_time = time.time()
        elapsed = current_time - self.start_time
        time_per_step = (current_time - self.last_log_time) / \
            max(1, metrics.get('steps_since_last_log', 1))
        self.last_log_time = current_time

        # 创建状态表格
        table = Table(
            title=f"训练进度 - Step {step}/{total_steps}", box=box.ROUNDED)
        table.add_column("指标", style="cyan", no_wrap=True)
        table.add_column("值", style="green", justify="right")

        # 基础信息
        table.add_row(
            "训练进度", f"{step}/{total_steps} ({100*step/total_steps:.1f}%)")
        table.add_row("环境步数", f"{env_steps:,}")
        table.add_row("已用时间", self._format_time(elapsed))
        table.add_row("每步耗时", f"{time_per_step:.3f}s")

        # 性能指标
        if 'episode_reward' in metrics:
            table.add_row("回合奖励", f"{metrics['episode_reward']:.3f}")
        if 'episode_length' in metrics:
            table.add_row("回合长度", f"{metrics['episode_length']:.0f}")

        # 训练指标
        if 'policy_loss' in metrics:
            table.add_row("策略损失", f"{metrics['policy_loss']:.4f}")
        if 'value_loss' in metrics:
            table.add_row("价值损失", f"{metrics['value_loss']:.4f}")
        if 'entropy' in metrics:
            table.add_row("熵", f"{metrics['entropy']:.4f}")
        if 'approx_kl' in metrics:
            table.add_row("近似KL", f"{metrics['approx_kl']:.4f}")
        if 'clip_fraction' in metrics:
            table.add_row("裁剪比例", f"{metrics['clip_fraction']:.2%}")

        self.console.print(table)

    def print_section(self, title: str):
        """打印分节标题

        Args:
            title: 标题文本
        """
        if self.use_rich and self.console:
            self.console.print()
            self.console.print(
                Panel(f"[bold cyan]{title}[/bold cyan]", box=box.DOUBLE))

    def print_summary(self, message: str, style: str = "green"):
        """打印总结信息

        Args:
            message: 消息内容
            style: 样式（green/red/yellow）
        """
        if self.use_rich and self.console:
            self.console.print()
            self.console.print(Panel.fit(
                f"[bold {style}]{message}[/bold {style}]",
                border_style=style
            ))

    def create_progress_bar(self, total: int, description: str = "训练中"):
        """创建Rich进度条

        Args:
            total: 总步数
            description: 描述文本

        Returns:
            Progress对象
        """
        if not self.use_rich:
            return None

        return Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=self.console,
        )

    def close(self):
        """关闭日志记录器"""
        if self.writer:
            self.writer.close()

    def _format_time(self, seconds: float) -> str:
        """格式化时间

        Args:
            seconds: 秒数

        Returns:
            格式化的时间字符串
        """
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)

        if hours > 0:
            return f"{hours}h {minutes}m {secs}s"
        elif minutes > 0:
            return f"{minutes}m {secs}s"
        else:
            return f"{secs}s"


class MetricsLogger:
    """指标累积器（用于多步平均）"""

    def __init__(self):
        self.metrics = {}
        self.counts = {}

    def log(self, key: str, value: float):
        """记录单个指标值

        Args:
            key: 指标名
            value: 指标值
        """
        if key not in self.metrics:
            self.metrics[key] = 0.0
            self.counts[key] = 0

        self.metrics[key] += float(value)
        self.counts[key] += 1

    def log_dict(self, metrics: Dict[str, float]):
        """批量记录指标

        Args:
            metrics: 指标字典
        """
        for key, value in metrics.items():
            self.log(key, value)

    def get_averages(self) -> Dict[str, float]:
        """获取平均值

        Returns:
            平均值字典
        """
        return {
            key: self.metrics[key] / self.counts[key]
            for key in self.metrics.keys()
        }

    def reset(self):
        """重置累积器"""
        self.metrics.clear()
        self.counts.clear()
