"""
日志系统：TensorBoard + Rich终端展示
"""

import os
import time
import jax.numpy as jp
from pathlib import Path
from typing import Dict, Any, Optional
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn, TimeElapsedColumn, ProgressColumn, Task
from rich.text import Text
from rich.live import Live
from rich.layout import Layout
from rich import box

# TensorBoard 支持（默认禁用，避免 torch 依赖冲突）
# 如需启用，设置环境变量: export ENABLE_TENSORBOARD=1
# 并安装: pip install torch --index-url https://download.pytorch.org/whl/cpu
HAS_TENSORBOARD = False
SummaryWriter = None

if os.environ.get("ENABLE_TENSORBOARD", "0") == "1":
    try:
        from torch.utils.tensorboard import SummaryWriter
        HAS_TENSORBOARD = True
    except ImportError:
        print("警告: ENABLE_TENSORBOARD=1 但未安装 torch，TensorBoard 日志已禁用")
        print("安装方法: pip install torch --index-url https://download.pytorch.org/whl/cpu")


class MetricsColumn(ProgressColumn):
    """自定义列：显示训练指标"""

    def render(self, task: Task) -> Text:
        """渲染训练指标"""
        if task.fields:
            metrics = []
            # 学习率
            if "lr" in task.fields:
                metrics.append(f"LR: {task.fields['lr']:.2e}")
            # 奖励
            if "reward" in task.fields:
                metrics.append(f"奖励: {task.fields['reward']:.2f}")
            # 损失
            if "loss" in task.fields:
                metrics.append(f"Loss: {task.fields['loss']:.4f}")
            # FPS
            if "fps" in task.fields:
                metrics.append(f"FPS: {task.fields['fps']:.0f}")

            if metrics:
                return Text(" | ".join(metrics), style="dim")

        return Text("")


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
        """创建Rich进度条（简单版本，仅进度条）

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
            TextColumn("•"),
            TextColumn("[cyan]步数: {task.completed}/{task.total}[/cyan]"),
            TimeElapsedColumn(),
            TextColumn("•"),
            TimeRemainingColumn(),
            console=self.console,
        )

    def create_training_display(self, total: int, steps_per_epoch: int = 1, description: str = "PPO训练"):
        """创建训练进度显示（多区域丰富布局）

        第一区：总轮次进度 + 当前步数进度 + 时间信息
        第二区：奖励与回合信息
        第三区：训练损失指标
        第四区：性能指标

        Args:
            total: 总训练轮次（epochs）
            steps_per_epoch: 每轮的步数
            description: 描述文本

        Returns:
            TrainingDisplay对象，提供update()方法更新显示
        """
        if not self.use_rich:
            return None

        return TrainingDisplay(
            console=self.console,
            total=total,
            steps_per_epoch=steps_per_epoch,
            description=description,
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


class TrainingDisplay:
    """训练进度显示（多区域丰富布局）

    使用Panel + Layout + Progress + Table实现分区显示：
    - 进度区：总轮次进度 + 当前步数进度 + 时间信息
    - 奖励区：回合奖励、回合长度等
    - 损失区：策略损失、价值损失、熵等
    - 性能区：FPS、SPS、学习率等
    """

    def __init__(self, console: Console, total: int, steps_per_epoch: int = 1, description: str = "PPO训练"):
        """初始化训练显示

        Args:
            console: Rich Console对象
            total: 总训练轮次（epochs）
            steps_per_epoch: 每轮的步数
            description: 任务描述
        """
        self.console = console
        self.total = total
        self.steps_per_epoch = steps_per_epoch
        self.description = description
        self.current_epoch = 0
        self.current_step = 0
        self.start_time = None
        self.warmup_steps = 2  # 跳过前2步（包含JIT编译）

        # 创建总轮次进度条
        self.epoch_progress = Progress(
            TextColumn("[bold cyan]总进度[/bold cyan]"),
            BarColumn(bar_width=30),
            TextColumn("[progress.percentage]{task.percentage:>3.1f}%"),
            TextColumn("[cyan]Epoch {task.completed}/{task.total}[/cyan]"),
            console=console,
        )

        # 创建当前步数进度条
        self.step_progress = Progress(
            TextColumn("[bold green]步进度[/bold green]"),
            BarColumn(bar_width=30),
            TextColumn("[progress.percentage]{task.percentage:>3.1f}%"),
            TextColumn("[green]Step {task.completed}/{task.total}[/green]"),
            console=console,
        )

        # 创建时间信息表格
        self.time_table = Table(
            show_header=False,
            show_edge=False,
            box=None,
            padding=(0, 1),
        )
        self.time_table.add_column("", style="dim", no_wrap=True)

        # 创建各个指标表格
        self.reward_table = self._create_metrics_table("奖励与回合")
        self.loss_table = self._create_metrics_table("训练损失")
        self.perf_table = self._create_metrics_table("性能指标")

        # 使用Layout组合所有组件
        self.layout = Layout()
        self.layout.split_column(
            Layout(name="progress", size=3),
            Layout(name="time", size=1),
            Layout(name="reward", size=3),
            Layout(name="loss", size=4),
            Layout(name="perf", size=3),
        )

        # 分配组件到各个区域
        self.layout["progress"].split_column(
            Layout(self.epoch_progress, size=1),
            Layout(self.step_progress, size=1),
        )
        self.layout["time"].update(self.time_table)
        self.layout["reward"].update(self.reward_table)
        self.layout["loss"].update(self.loss_table)
        self.layout["perf"].update(self.perf_table)

        # 创建外层Panel
        self.panel = Panel(
            self.layout,
            title=f"[bold magenta]{description} 监控面板[/bold magenta]",
            border_style="magenta",
            box=box.DOUBLE,
        )

        # 创建Live显示
        self.live = Live(
            self.panel,
            console=console,
            refresh_per_second=4,
            transient=False,
        )

        self.epoch_task_id = None
        self.step_task_id = None
        self.started = False

    def _create_metrics_table(self, title: str) -> Table:
        """创建指标表格

        Args:
            title: 表格标题

        Returns:
            Table对象
        """
        table = Table(
            title=f"[bold]{title}[/bold]",
            show_header=False,
            show_edge=False,
            box=box.SIMPLE,
            padding=(0, 1),
        )
        table.add_column("", style="dim", no_wrap=True, width=20)
        table.add_column("", justify="right", style="bold cyan", width=10)
        table.add_column("", style="dim", no_wrap=True, width=20)
        table.add_column("", justify="right", style="bold green", width=10)
        return table

    def start(self):
        """启动显示"""
        if not self.started:
            self.live.start()
            self.epoch_task_id = self.epoch_progress.add_task(
                "",
                total=self.total
            )
            self.step_task_id = self.step_progress.add_task(
                "",
                total=self.steps_per_epoch
            )
            self.started = True
            import time
            self.start_time = time.time()

    def update(self, epoch: Optional[int] = None, step: Optional[int] = None, metrics: Optional[Dict[str, Any]] = None):
        """更新显示

        Args:
            epoch: 当前轮次（如果提供，会更新总进度）
            step: 当前步数（如果提供，会更新步进度）
            metrics: 当前训练指标字典
        """
        if not self.started:
            self.start()

        # 更新轮次进度
        if epoch is not None:
            advance_epoch = epoch - self.current_epoch
            if advance_epoch > 0:
                self.current_epoch = epoch
                self.epoch_progress.update(self.epoch_task_id, completed=epoch)

        # 更新步数进度
        if step is not None:
            # 如果step回到0，说明新的epoch开始，重置step进度
            if step < self.current_step:
                self.step_progress.reset(self.step_task_id)
            self.current_step = step
            self.step_progress.update(self.step_task_id, completed=step)

        # 更新时间信息
        self._update_time_table()

        # 更新指标表格
        if metrics:
            self._update_metrics_tables(metrics)

        # 刷新显示
        self.live.refresh()

    def _update_time_table(self):
        """更新时间信息表格"""
        if self.start_time is None:
            return

        import time
        current_time = time.time()
        elapsed = current_time - self.start_time

        # 计算预计剩余时间
        if self.current_epoch > self.warmup_steps and self.total > 0:
            time_per_epoch = elapsed / max(1, self.current_epoch - self.warmup_steps)
            remaining = time_per_epoch * (self.total - self.current_epoch)
            eta_seconds = int(current_time + remaining)
            eta_time = time.strftime("%H:%M:%S", time.localtime(eta_seconds))
            time_info = f"[dim]已用[/dim] {self._format_time(elapsed)} [dim]|[/dim] [dim]剩余[/dim] {self._format_time(remaining)} [dim]| ETA[/dim] {eta_time}"
        else:
            time_info = f"[dim]已用[/dim] {self._format_time(elapsed)} [dim]| 预热中...[/dim]"

        # 重建表格
        self.time_table = Table(
            show_header=False,
            show_edge=False,
            box=None,
            padding=(0, 1),
        )
        self.time_table.add_column("", style="dim", no_wrap=True)
        self.time_table.add_row(time_info)
        self.layout["time"].update(self.time_table)

    def _update_metrics_tables(self, metrics: Dict[str, Any]):
        """更新所有指标表格

        Args:
            metrics: 指标字典
        """
        # 更新奖励表格
        self.reward_table = self._create_metrics_table("奖励与回合")
        if "episode_reward" in metrics:
            self.reward_table.add_row(
                "回合奖励:", f"{metrics['episode_reward']:.2f}",
                "回合长度:", f"{int(metrics.get('episode_length', 0))}"
            )
        if "best_reward" in metrics:
            self.reward_table.add_row(
                "最佳奖励:", f"{metrics['best_reward']:.2f}",
                "平均长度:", f"{int(metrics.get('avg_length', 0))}"
            )
        self.layout["reward"].update(self.reward_table)

        # 更新损失表格
        self.loss_table = self._create_metrics_table("训练损失")
        if "policy_loss" in metrics:
            self.loss_table.add_row(
                "策略损失:", f"{metrics['policy_loss']:.4f}",
                "价值损失:", f"{metrics.get('value_loss', 0):.4f}"
            )
        if "entropy" in metrics:
            self.loss_table.add_row(
                "熵:", f"{metrics['entropy']:.4f}",
                "近似KL:", f"{metrics.get('approx_kl', 0):.4f}"
            )
        if "clip_fraction" in metrics:
            self.loss_table.add_row(
                "裁剪比例:", f"{metrics['clip_fraction']:.2%}",
                "解释方差:", f"{metrics.get('explained_variance', 0):.2f}"
            )
        self.layout["loss"].update(self.loss_table)

        # 更新性能表格
        self.perf_table = self._create_metrics_table("性能指标")
        if "fps" in metrics:
            self.perf_table.add_row(
                "FPS:", f"{int(metrics['fps'])}",
                "SPS:", f"{int(metrics.get('sps', 0))}"
            )
        if "learning_rate" in metrics:
            self.perf_table.add_row(
                "学习率:", f"{metrics['learning_rate']:.2e}",
                "梯度范数:", f"{metrics.get('grad_norm', 0):.3f}"
            )
        self.layout["perf"].update(self.perf_table)

    def stop(self):
        """停止显示"""
        if self.started:
            self.live.stop()
            self.started = False

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

    def __enter__(self):
        """上下文管理器入口"""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.stop()
