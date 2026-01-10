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

# ============================================================================================
# ======================================= Rich组件 ============================================
# ============================================================================================


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


# ============================================================================================
# ===================================== END: Rich组件 ==========================================
# ============================================================================================


# ============================================================================================
# ======================================= 日志记录器 ==========================================
# ============================================================================================


class Logger:
    """TensorBoard日志记录器"""

    def __init__(
        self,
        log_dir: str = "logs",
        use_tensorboard: bool = True,
        use_rich: bool = True,
    ):
        """初始化日志记录器"""
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.use_rich = use_rich

        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter

                self.writer = SummaryWriter(log_dir=str(self.log_dir), flush_secs=10)
                self.disabled = False
            except Exception as torch_exc:
                try:
                    from tensorboard.compat.proto import event_pb2, summary_pb2
                    from tensorboard.summary.writer.event_file_writer import EventFileWriter

                    class _TBEventWriter:
                        def __init__(self, log_dir: Path, flush_secs: float = 10.0):
                            self._writer = EventFileWriter(
                                str(log_dir),
                                flush_secs=flush_secs,
                            )

                        def add_scalar(self, tag: str, scalar_value: float, global_step: int):
                            summary = summary_pb2.Summary(
                                value=[
                                    summary_pb2.Summary.Value(
                                        tag=tag,
                                        simple_value=float(scalar_value),
                                    )
                                ]
                            )
                            event = event_pb2.Event(
                                wall_time=time.time(),
                                step=int(global_step),
                                summary=summary,
                            )
                            self._writer.add_event(event)

                        def close(self):
                            self._writer.close()

                    self.writer = _TBEventWriter(self.log_dir, flush_secs=10.0)
                    self.disabled = False
                except Exception as tb_exc:
                    self.writer = None
                    self.disabled = True
                    print(
                        "警告: TensorBoard日志将被禁用（缺少写入依赖）。"
                        f" torch.utils.tensorboard 导入错误: {torch_exc};"
                        f" tensorboard EventFileWriter 导入错误: {tb_exc}"
                    )
        else:
            self.writer = None
            self.disabled = True

    # --------------------------------------------------------------------------------------------

    def log_scalar(self, name: str, value: float, step: int):
        """记录标量"""
        if not self.disabled and self.writer is not None:
            self.writer.add_scalar(name, value, step)

    def log_scalars(self, metrics: Dict[str, float], step: int, prefix: str = ""):
        """批量记录标量"""
        if not self.disabled and self.writer is not None:
            for name, value in metrics.items():
                if isinstance(value, (int, float)):
                    full_name = f"{prefix}/{name}" if prefix else name
                    self.writer.add_scalar(full_name, value, step)

    # --------------------------------------------------------------------------------------------

    def close(self):
        """关闭日志记录器"""
        if not self.disabled and self.writer is not None:
            self.writer.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# ============================================================================================
# ===================================== END: 日志记录器 =======================================
# ============================================================================================


# ============================================================================================
# ======================================= 指标统计器 ==========================================
# ============================================================================================


class MetricsLogger:
    """指标统计器（带Episode统计）"""

    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.metrics = {}
        self.counts = {}

    def log_dict(self, metrics: Dict[str, Any], weight: float = 1.0):
        """记录指标字典"""
        for key, value in metrics.items():
            if isinstance(value, (int, float, jp.ndarray)):
                if isinstance(value, jp.ndarray):
                    value = float(value)
                self._log(key, value, weight)

    def _log(self, key: str, value: float, weight: float):
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


# ============================================================================================
# ===================================== END: 指标统计器 =======================================
# ============================================================================================


# ============================================================================================
# ======================================= 训练显示 ============================================
# ============================================================================================


class TrainingDisplay:
    """训练进度显示（参考 rsl_rl 格式）"""

    def __init__(
        self,
        console: Console,
        total: int,
        steps_per_epoch: int = 1,
        description: str = "PPO训练",
        reward_columns: int = 2,
    ):
        self.console = console
        self.total = total
        self.description = description
        self.current_epoch = 0
        self.start_time = None
        self.warmup_steps = 2
        self.current_metrics = {}
        self.reward_columns = max(1, int(reward_columns))

        self.progress_bar = Progress(
            TextColumn("  "),
            BarColumn(bar_width=60),
            TextColumn("[progress.percentage]{task.percentage:>3.1f}%"),
            TextColumn("•"),
            TextColumn("{task.completed}/{task.total}"),
            console=console,
        )

        self.live = Live(
            self._build_content(),
            console=console,
            refresh_per_second=4,
        )

        self.progress_task_id = None
        self.started = False

    # --------------------------------------------------------------------------------------------

    def _build_content(self):
        """构建显示内容"""
        summary_text, reward_items, other_items = self._format_details_structured()

        renderables = [self.progress_bar, Text(""), Text(summary_text)]

        if other_items:
            renderables.append(Text(""))
            renderables.append(self._render_kv_table("Details", other_items, columns=1))

        if reward_items:
            renderables.append(Text(""))
            renderables.append(
                self._render_kv_table(
                    "Rewards", reward_items, columns=self.reward_columns
                )
            )

        return Group(*renderables)

    def _render_kv_table(self, title: str, items: list[tuple[str, float]], columns: int):
        columns = max(1, int(columns))
        grid = Table.grid(padding=(0, 2))
        for _ in range(columns):
            grid.add_column(justify="left", ratio=1)
            grid.add_column(justify="right", width=12)

        def _pad_row(row: list[str]) -> list[str]:
            target = columns * 2
            if len(row) < target:
                row.extend(["", ""] * ((target - len(row)) // 2))
            return row

        for i in range(0, len(items), columns):
            row = []
            for name, value in items[i : i + columns]:
                row.append(str(name))
                row.append(f"{float(value):.4f}")
            grid.add_row(*_pad_row(row))

        return Panel(grid, title=title, border_style="dim", box=box.ROUNDED)

    def _format_details_structured(self):
        """格式化详情信息（结构化，便于多列展示）"""
        lines: list[str] = []

        if self.start_time:
            import time

            elapsed = time.time() - self.start_time
            if self.current_epoch > self.warmup_steps and self.total > 0:
                time_per_epoch = elapsed / max(
                    1, self.current_epoch - self.warmup_steps
                )
                remaining = time_per_epoch * (self.total - self.current_epoch)
            else:
                remaining = None

            lines.append(f"{ 'Time elapsed:':<30}{self._format_hms(elapsed)}")
            if remaining is not None:
                lines.append(f"{ 'ETA:':<30}{self._format_hms(remaining)}")

        if not self.current_metrics:
            if self.current_epoch == 0:
                lines.append(f"\n{ 'Status:':<30}正在初始化训练...")
            else:
                lines.append(f"\n{ 'Status:':<30}正在训练中...")
            lines.append(
                f"{ 'Current iteration:':<30}{self.current_epoch}/{self.total}"
            )
            return "\n".join(lines), [], []

        env_steps = self._get_value("env_steps", "perf/total_env_steps")
        if env_steps is not None:
            lines.append(f"{ 'Total steps:':<30}{int(env_steps):,}")

        sps = self._get_value("sps", "perf/steps_per_sec", "perf/avg_steps_per_sec")
        if sps is not None:
            lines.append(f"{ 'Steps per second:':<30}{int(sps)}")

        if "collection_time" in self.current_metrics:
            lines.append(
                f"{ 'Collection time:':<30}{self.current_metrics['collection_time']:.3f}s"
            )
        if "learning_time" in self.current_metrics:
            lines.append(
                f"{ 'Learning time:':<30}{self.current_metrics['learning_time']:.3f}s"
            )

        lines.append("")

        value_loss = self._get_value("value_loss", "train/value_loss")
        if value_loss is not None:
            lines.append(f"{ 'Mean value loss:':<30}{value_loss:.4f}")

        policy_loss = self._get_value(
            "surrogate_loss", "policy_loss", "train/surrogate_loss"
        )
        if policy_loss is not None:
            lines.append(f"{ 'Mean policy loss:':<30}{policy_loss:.4f}")

        entropy = self._get_value("entropy_loss", "entropy", "train/entropy_loss")
        if entropy is not None:
            lines.append(f"{ 'Mean entropy loss:':<30}{entropy:.4f}")

        kl = self._get_value("approx_kl", "kl", "train/approx_kl")
        if kl is not None:
            lines.append(f"{ 'Mean KL divergence:':<30}{kl:.4f}")

        clip_frac = self._get_value("clip_fraction", "train/clip_fraction")
        if clip_frac is not None:
            lines.append(f"{ 'Clip fraction:':<30}{clip_frac:.2%}")

        lines.append("")

        episode_reward = self._get_value(
            "episode_reward", "train/episode_reward", "mean_reward"
        )
        if episode_reward is not None:
            lines.append(f"{ 'Mean reward:':<30}{episode_reward:.2f}")

        episode_length = self._get_value("episode_length", "train/episode_length")
        if episode_length is not None:
            lines.append(f"{ 'Mean episode length:':<30}{episode_length:.1f}")

        lr = self._get_value("learning_rate", "train/learning_rate")
        if lr is not None:
            lines.append(f"{ 'Learning rate:':<30}{lr:.6f}")

        reward_items: list[tuple[str, float]] = []
        other_items: list[tuple[str, float]] = []
        for key in sorted(self.current_metrics.keys()):
            if any(
                key.startswith(prefix)
                for prefix in [
                    "Episode_Reward/",
                    "Metrics/",
                    "Curriculum/",
                    "Episode_Termination/",
                    "reward/",
                    "train/Episode_Reward/",
                    "train/Metrics/",
                    "train/Curriculum/",
                    "train/Episode_Termination/",
                    "train/reward/",
                ]
            ):
                if not any(
                    x in key
                    for x in ["episode_reward", "episode_length", "action_noise_std"]
                ):
                    value = self.current_metrics[key]
                    display_key = key
                    is_reward = False
                    for prefix in [
                        "train/Episode_Reward/",
                        "train/Metrics/",
                        "train/Curriculum/",
                        "train/Episode_Termination/",
                        "train/reward/",
                        "Episode_Reward/",
                        "Metrics/",
                        "Curriculum/",
                        "Episode_Termination/",
                        "reward/",
                    ]:
                        if key.startswith(prefix):
                            display_key = key[len(prefix) :]
                            if prefix in {"train/reward/", "reward/"}:
                                is_reward = True
                    display_key = display_key.replace("_", " ")
                    if isinstance(value, (int, float, jp.ndarray)):
                        if isinstance(value, jp.ndarray):
                            value = float(value)
                        if is_reward:
                            reward_items.append((display_key, float(value)))
                        else:
                            other_items.append((display_key, float(value)))

        return "\n".join(lines), reward_items, other_items

    # --------------------------------------------------------------------------------------------

    def _get_value(self, *keys):
        for key in keys:
            if key in self.current_metrics:
                return self.current_metrics[key]
        return None

    def _format_hms(self, seconds: float) -> str:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    def start(self):
        if not self.started:
            self.live.start()
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
        if not self.started:
            self.start()
        if epoch is not None:
            if epoch > self.current_epoch:
                self.current_epoch = epoch
                self.progress_bar.update(self.progress_task_id, completed=epoch)
        if metrics:
            self.current_metrics = metrics.copy()
        self.live.update(self._build_content())

    def stop(self):
        if self.started:
            self.live.stop()
            self.started = False

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


# ============================================================================================
# ===================================== END: 训练显示 =========================================
# ============================================================================================


# ============================================================================================
# ======================================= 便捷函数 ============================================
# ============================================================================================


def create_logger(log_dir: str = "logs") -> Logger:
    """创建日志记录器"""
    return Logger(log_dir=log_dir)


# --------------------------------------------------------------------------------------------


def create_training_display(
    console: Console,
    total: int,
    steps_per_epoch: int = 1,
    description: str = "PPO训练",
) -> TrainingDisplay:
    """创建训练显示"""
    return TrainingDisplay(
        console=console,
        total=total,
        steps_per_epoch=steps_per_epoch,
        description=description,
    )


# --------------------------------------------------------------------------------------------


def print_summary(message: str, style: str = "cyan", console: Optional[Console] = None):
    """打印总结信息"""
    if console is None:
        console = Console()
    console.print(f"[{style}]{message}[/{style}]")


# ============================================================================================
# ===================================== END: 便捷函数 ==========================================
# ============================================================================================
