"""
日志系统：TensorBoard + Rich终端展示
"""

import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

import jax.numpy as jp
from rich import box
from rich.align import Align
from rich.columns import Columns
from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress import (BarColumn, Progress, ProgressColumn, SpinnerColumn,
                           Task, TextColumn, TimeElapsedColumn,
                           TimeRemainingColumn)
from rich.table import Table
from rich.text import Text

# ==================== Rich进度列定制 ====================


class MetricsColumn(ProgressColumn):
    """显示自定义指标"""

    def __init__(self, metric_name: str, format_spec: str = ".2f"):
        self.metric_name = metric_name
        self.format_spec = format_spec
        super().__init__()

    def render(self, task: Task) -> Text:
        if self.metric_name in task.fields:
            value = task.fields[self.metric_name]
            return Text(f"{value:{self.format_spec}}", style="cyan")
        return Text("")


# ==================== TensorBoard日志 ====================


class Logger:
    """TensorBoard日志记录器"""

    def __init__(
        self, log_dir: str = "logs", use_tensorboard: bool = True, use_rich: bool = True
    ):
        """初始化日志记录器

        Args:
            log_dir: 日志保存目录
            use_tensorboard: 是否使用TensorBoard（保留参数以兼容）
            use_rich: 是否使用Rich终端输出（保留参数以兼容）
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.use_rich = use_rich

        # 尝试导入TensorBoard
        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter

                self.writer = SummaryWriter(log_dir=str(self.log_dir), flush_secs=10)
                self.disabled = False
            except ImportError:
                self.writer = None
                self.disabled = True
                print(f"警告: TensorBoard未安装，日志将被禁用")
        else:
            self.writer = None
            self.disabled = True

    def log_scalar(self, name: str, value: float, step: int):
        """记录标量

        Args:
            name: 指标名称
            value: 指标值
            step: 训练步数
        """
        if not self.disabled and self.writer is not None:
            self.writer.add_scalar(name, value, step)

    def log_scalars(self, metrics: Dict[str, float], step: int, prefix: str = ""):
        """批量记录标量

        Args:
            metrics: 指标字典
            step: 训练步数
            prefix: 名称前缀
        """
        if not self.disabled and self.writer is not None:
            for name, value in metrics.items():
                if isinstance(value, (int, float)):
                    full_name = f"{prefix}/{name}" if prefix else name
                    self.writer.add_scalar(full_name, value, step)

    def close(self):
        """关闭日志记录器"""
        if not self.disabled and self.writer is not None:
            self.writer.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# ==================== 指标统计（类rsl_rl实现）====================


class MetricsLogger:
    """指标统计器（带Episode统计）- 参考rsl_rl实现"""

    def __init__(self, window_size: int = 100):
        """初始化指标统计器

        Args:
            window_size: 滑动窗口大小
        """
        self.window_size = window_size
        from collections import deque

        self.metrics = {}
        self.counts = {}

    def log_dict(self, metrics: Dict[str, Any], weight: float = 1.0):
        """记录指标字典

        Args:
            metrics: 指标字典
            weight: 权重（用于加权平均）
        """
        for key, value in metrics.items():
            if isinstance(value, (int, float, jp.ndarray)):
                # 转换JAX数组为Python标量
                if isinstance(value, jp.ndarray):
                    value = float(value)
                self._log(key, value, weight)

    def _log(self, key: str, value: float, weight: float):
        """记录单个指标"""
        if key not in self.metrics:
            self.metrics[key] = value * weight
            self.counts[key] = weight
        else:
            self.metrics[key] += value * weight
            self.counts[key] += weight

    def get_averages(self) -> Dict[str, float]:
        """获取平均值"""
        return {key: self.metrics[key] / self.counts[key] for key in self.metrics}

    def reset(self):
        """重置统计"""
        self.metrics.clear()
        self.counts.clear()


# ==================== 训练进度显示（参考rsl_rl格式）====================


class TrainingDisplay:
    """训练进度显示（参考 rsl_rl 格式）

    格式：
    - 上方：详细信息（性能、损失、奖励等）
    - 下方：Rich 进度条（无边框）
    """

    def __init__(
        self,
        console: Console,
        total: int,
        steps_per_epoch: int = 1,
        description: str = "PPO训练",
    ):
        """初始化训练显示

        Args:
            console: Rich Console对象
            total: 总训练轮次（epochs）
            steps_per_epoch: 每轮的步数（保留参数但不再使用）
            description: 任务描述
        """
        self.console = console
        self.total = total
        self.description = description
        self.current_epoch = 0
        self.start_time = None
        self.warmup_steps = 2  # 跳过前2步（包含JIT编译）
        self.current_metrics = {}

        # 创建进度条（放在底部）
        self.progress_bar = Progress(
            TextColumn("  "),
            BarColumn(bar_width=60),
            TextColumn("[progress.percentage]{task.percentage:>3.1f}%"),
            TextColumn("•"),
            TextColumn("{task.completed}/{task.total}"),
            console=console,
        )

        # 创建Live显示
        self.live = Live(
            self._build_content(),
            console=console,
            refresh_per_second=4,
        )

        self.progress_task_id = None
        self.started = False

    def _build_content(self):
        """构建完整的显示内容（上方详情，下方进度条）"""
        # 详情文本
        details_text = self._format_details()

        return Group(
            Text(details_text),
            Text(""),
            self.progress_bar,
        )

    def _format_details(self):
        """格式化详情信息（参考 rsl_rl 格式）"""
        lines = []

        # ==================== 性能指标 ====================
        if self.start_time:
            import time

            elapsed = time.time() - self.start_time

            # ETA
            if self.current_epoch > self.warmup_steps and self.total > 0:
                time_per_epoch = elapsed / max(
                    1, self.current_epoch - self.warmup_steps
                )
                remaining = time_per_epoch * (self.total - self.current_epoch)
            else:
                remaining = None

            lines.append(f"{'Time elapsed:':<30}{self._format_hms(elapsed)}")
            if remaining is not None:
                lines.append(f"{'ETA:':<30}{self._format_hms(remaining)}")

        # 如果没有指标数据，显示基本状态
        if not self.current_metrics:
            if self.current_epoch == 0:
                lines.append(f"\n{'Status:':<30}正在初始化训练...")
            else:
                lines.append(f"\n{'Status:':<30}正在训练中...")
            lines.append(f"{'Current iteration:':<30}{self.current_epoch}/{self.total}")
            return "\n".join(lines)

        # 总步数
        env_steps = self._get_value("env_steps", "perf/total_env_steps")
        if env_steps is not None:
            lines.append(f"{'Total steps:':<30}{int(env_steps):,}")

        # SPS
        sps = self._get_value("sps", "perf/steps_per_sec", "perf/avg_steps_per_sec")
        if sps is not None:
            lines.append(f"{'Steps per second:':<30}{int(sps)}")

        # Collection/Learning time
        if "collection_time" in self.current_metrics:
            lines.append(
                f"{'Collection time:':<30}{self.current_metrics['collection_time']:.3f}s"
            )
        if "learning_time" in self.current_metrics:
            lines.append(
                f"{'Learning time:':<30}{self.current_metrics['learning_time']:.3f}s"
            )

        lines.append("")  # 空行

        # ==================== 损失指标 ====================
        value_loss = self._get_value("value_loss", "train/value_loss")
        if value_loss is not None:
            lines.append(f"{'Mean value loss:':<30}{value_loss:.4f}")

        policy_loss = self._get_value(
            "surrogate_loss", "policy_loss", "train/surrogate_loss"
        )
        if policy_loss is not None:
            lines.append(f"{'Mean policy loss:':<30}{policy_loss:.4f}")

        entropy = self._get_value("entropy_loss", "entropy", "train/entropy_loss")
        if entropy is not None:
            lines.append(f"{'Mean entropy loss:':<30}{entropy:.4f}")

        kl = self._get_value("approx_kl", "kl", "train/approx_kl")
        if kl is not None:
            lines.append(f"{'Mean KL divergence:':<30}{kl:.4f}")

        clip_frac = self._get_value("clip_fraction", "train/clip_fraction")
        if clip_frac is not None:
            lines.append(f"{'Clip fraction:':<30}{clip_frac:.2%}")

        lines.append("")  # 空行

        # ==================== 奖励与统计 ====================
        episode_reward = self._get_value(
            "episode_reward", "train/episode_reward", "mean_reward"
        )
        if episode_reward is not None:
            lines.append(f"{'Mean reward:':<30}{episode_reward:.2f}")

        episode_length = self._get_value("episode_length", "train/episode_length")
        if episode_length is not None:
            lines.append(f"{'Mean episode length:':<30}{episode_length:.1f}")

        lr = self._get_value("learning_rate", "train/learning_rate")
        if lr is not None:
            lines.append(f"{'Learning rate:':<30}{lr:.6f}")

        # ==================== 详细指标 ====================
        # 收集其他详细指标
        detail_lines = []
        for key in sorted(self.current_metrics.keys()):
            if any(
                key.startswith(prefix)
                for prefix in [
                    "Episode_Reward/",
                    "Metrics/",
                    "Curriculum/",
                    "Episode_Termination/",
                    "train/Episode_Reward/",
                    "train/Metrics/",
                    "train/Curriculum/",
                    "train/Episode_Termination/",
                ]
            ):
                # 跳过已显示的核心指标
                if not any(
                    x in key
                    for x in ["episode_reward", "episode_length", "action_noise_std"]
                ):
                    value = self.current_metrics[key]
                    display_key = key
                    for prefix in [
                        "train/Episode_Reward/",
                        "train/Metrics/",
                        "train/Curriculum/",
                        "train/Episode_Termination/",
                        "Episode_Reward/",
                        "Metrics/",
                        "Curriculum/",
                        "Episode_Termination/",
                    ]:
                        if key.startswith(prefix):
                            display_key = key[len(prefix) :]
                    display_key = display_key.replace("_", " ")
                    detail_lines.append(f"{display_key:<30}{float(value):.4f}")

        if detail_lines:
            lines.append("")
            lines.extend(detail_lines)

        return "\n".join(lines)

    def _get_value(self, *keys):
        """获取指标值，支持多种键名格式"""
        for key in keys:
            if key in self.current_metrics:
                return self.current_metrics[key]
        return None

    def _format_hms(self, seconds: float) -> str:
        """格式化时间为 HH:MM:SS"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    def start(self):
        """启动显示"""
        if not self.started:
            self.live.start()
            # 添加进度条任务
            self.progress_task_id = self.progress_bar.add_task("", total=self.total)
            self.started = True
            import time

            self.start_time = time.time()

    def update(
        self,
        epoch: Optional[int] = None,
        step: Optional[int] = None,
        metrics: Optional[Dict[str, Any]] = None,
    ):
        """更新显示

        Args:
            epoch: 当前轮次（如果提供，会更新总进度）
            step: 当前步数（保留参数但不再使用）
            metrics: 当前训练指标字典
        """
        if not self.started:
            self.start()

        # 更新轮次进度
        if epoch is not None:
            if epoch > self.current_epoch:
                self.current_epoch = epoch
                self.progress_bar.update(self.progress_task_id, completed=epoch)

        # 更新指标
        if metrics:
            self.current_metrics = metrics.copy()

        # 更新Live显示内容
        self.live.update(self._build_content())

    def stop(self):
        """停止显示"""
        if self.started:
            self.live.stop()
            self.started = False

    def __enter__(self):
        """上下文管理器入口"""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.stop()


# ==================== 便捷函数 ====================


def create_logger(log_dir: str = "logs") -> Logger:
    """创建日志记录器

    Args:
        log_dir: 日志保存目录

    Returns:
        Logger实例
    """
    return Logger(log_dir=log_dir)


def create_training_display(
    console: Console, total: int, steps_per_epoch: int = 1, description: str = "PPO训练"
) -> TrainingDisplay:
    """创建训练显示

    Args:
        console: Rich Console对象
        total: 总训练轮次
        steps_per_epoch: 每轮步数
        description: 任务描述

    Returns:
        TrainingDisplay实例
    """
    return TrainingDisplay(
        console=console,
        total=total,
        steps_per_epoch=steps_per_epoch,
        description=description,
    )


def print_summary(message: str, style: str = "cyan", console: Optional[Console] = None):
    """打印总结信息

    Args:
        message: 消息内容
        style: Rich样式
        console: Console对象（可选）
    """
    if console is None:
        console = Console()
    console.print(f"[{style}]{message}[/{style}]")
