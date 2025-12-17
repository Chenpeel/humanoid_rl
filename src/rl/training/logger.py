"""
日志系统：TensorBoard + Rich终端展示
"""

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

    def create_training_display(self, total: int, description: str = "PPO训练"):
        """创建训练进度显示（两行刷新显示）

        第一行：进度条 + epoch/steps + 时间
        第二行：训练指标（reward, loss, FPS等）- 动态刷新

        Args:
            total: 总训练步数
            description: 描述文本

        Returns:
            TrainingDisplay对象，提供update()方法更新显示
        """
        if not self.use_rich:
            return None

        return TrainingDisplay(
            console=self.console,
            total=total,
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
    """训练进度显示（两行动态刷新）

    第一行：进度条 + step/steps + 时间估计
    第二行：训练指标表格（动态刷新，不重复打印）
    """

    def __init__(self, console: Console, total: int, description: str = "PPO训练"):
        """初始化训练显示

        Args:
            console: Rich Console对象
            total: 总训练步数
            description: 任务描述
        """
        self.console = console
        self.total = total
        self.description = description
        self.current = 0
        self.start_time = None  # 用于计算准确的ETA
        self.warmup_steps = 2   # 跳过前2步（包含JIT编译）

        # 创建进度条（第一行）
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.1f}%"),
            TextColumn("[cyan]{task.completed}/{task.total}[/cyan]"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console,
            transient=False,  # 训练结束后保留进度条
        )

        # 创建指标表格（第二行）
        self.metrics_table = Table(
            show_header=False,
            show_edge=False,
            box=None,
            padding=(0, 1),
        )
        self.metrics_table.add_column("", style="dim", no_wrap=True)
        self.metrics_table.add_column("", justify="right", style="bold")

        # 使用Layout组合两个组件
        from rich.layout import Layout
        self.layout = Layout()
        self.layout.split_column(
            Layout(self.progress, name="progress", size=1),
            Layout(self.metrics_table, name="metrics", size=1),
        )

        # 创建Live显示
        self.live = Live(
            self.layout,
            console=console,
            refresh_per_second=4,  # 每秒刷新4次
            transient=False,  # 训练结束后保留显示
        )

        self.task_id = None
        self.started = False

    def start(self):
        """启动显示"""
        if not self.started:
            self.live.start()
            self.task_id = self.progress.add_task(
                f"[cyan]{self.description}",
                total=self.total
            )
            self.started = True

    def update(self, advance: int = 1, metrics: Optional[Dict[str, Any]] = None):
        """更新显示

        Args:
            advance: 进度增加量
            metrics: 当前训练指标字典
        """
        if not self.started:
            self.start()

        # 更新进度条
        self.current += advance

        # ✅ 修复ETA：跳过warmup步骤，从第3步开始计时
        if self.current == self.warmup_steps:
            # 重置进度条的时间基准（排除编译时间）
            import time
            self.start_time = time.time()
            # 重新创建task，重置起始时间
            self.progress.remove_task(self.task_id)
            self.task_id = self.progress.add_task(
                f"[cyan]{self.description}",
                total=self.total,
                completed=self.current
            )
        else:
            self.progress.update(self.task_id, advance=advance)

        # 更新指标表格
        if metrics:
            self._update_metrics_table(metrics)

        # 刷新显示
        self.live.refresh()

    def _update_metrics_table(self, metrics: Dict[str, Any]):
        """更新指标表格

        Args:
            metrics: 指标字典
        """
        # 清空表格
        self.metrics_table = Table(
            show_header=False,
            show_edge=False,
            box=None,
            padding=(0, 1),
        )
        self.metrics_table.add_column("", style="dim", no_wrap=True)
        self.metrics_table.add_column("", justify="right", style="bold")

        # 定义显示顺序和格式
        display_keys = [
            ("episode_reward", "回合奖励", ".3f"),
            ("episode_length", "回合长度", ".0f"),
            ("policy_loss", "策略损失", ".4f"),
            ("value_loss", "价值损失", ".4f"),
            ("entropy", "熵", ".4f"),
            ("approx_kl", "近似KL", ".4f"),
            ("clip_fraction", "裁剪比例", ".2%"),
            ("learning_rate", "学习率", ".2e"),
            ("fps", "FPS", ".0f"),
            ("sps", "SPS", ".0f"),
        ]

        # 构建表格行
        row_items = []
        for key, label, fmt in display_keys:
            if key in metrics:
                value = metrics[key]
                if fmt.endswith('%'):
                    # 百分比格式
                    formatted = f"{value:{fmt}}"
                else:
                    formatted = f"{value:{fmt}}"
                row_items.append(f"[dim]{label}:[/dim] [green]{formatted}[/green]")

        # 每行显示4个指标，分成多行
        items_per_row = 4
        for i in range(0, len(row_items), items_per_row):
            row = row_items[i:i+items_per_row]
            self.metrics_table.add_row(" | ".join(row))

        # 更新Layout中的metrics部分
        self.layout["metrics"].update(self.metrics_table)

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
