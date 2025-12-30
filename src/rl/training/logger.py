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
from rich.console import Group
from rich.columns import Columns
from rich.align import Align
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
    """训练进度显示（无边框设计）

    使用 Live + Group 实现无边框显示：
    - 标题区：居中显示任务名称
    - 进度区：Rich进度条
    - 内容区：简洁的文本格式指标，无底边框
    """

    def __init__(self, console: Console, total: int, steps_per_epoch: int = 1, description: str = "PPO训练"):
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
        self.metrics_text = ""

        # 创建总轮次进度条（使用独立的进度条对象）
        self.progress_renderable = Progress(
            TextColumn("  "),
            BarColumn(bar_width=50),
            TextColumn("[progress.percentage]{task.percentage:>3.1f}%"),
            console=console,
            expand=False,
        )

        # 创建Live显示（初始内容为空）
        self.live = Live(
            self._build_content(),
            console=console,
            refresh_per_second=4,
        )

        self.epoch_task_id = None
        self.progress_task_id = None
        self.started = False

    def _build_content(self):
        """构建完整的显示内容"""
        # 标题行
        title_line = self._center_text(self.description, 78)

        # 进度条描述
        progress_desc = f"Learning iteration {self.current_epoch}/{self.total}"

        # 分隔线
        separator = "├" + "─" * 76 + "┤"

        return Group(
            Text("┌" + "─" * 76 + "┐", style="cyan"),
            Text("│" + title_line + "│", style="cyan bold"),
            Text(separator, style="cyan"),
            Text(f"│ {progress_desc:<76}│", style="cyan"),
            # 进度条（嵌入到边框中）
            self._embed_progress_in_border(),
            Text(separator, style="cyan"),
            # 指标内容（无底边框）
            self._render_metrics_lines(),
        )

    def _center_text(self, text: str, width: int) -> str:
        """居中文本"""
        padding = (width - len(text)) // 2
        return " " * padding + text + " " * (width - padding - len(text))

    def _embed_progress_in_border(self):
        """将进度条嵌入到边框中"""
        # 进度条行：使用 Columns 实现水平布局
        from rich.columns import Columns
        from rich.align import Align

        # 居中的进度条
        centered_progress = Align.center(
            self.progress_renderable,
            width=72,  # 留出边框空间
        )

        # 使用 Columns 水平排列：左边框、空格、进度条、空格、右边框
        return Columns(
            [
                Text("│", style="cyan"),
                Text(" ", style=""),
                centered_progress,
                Text(" ", style=""),
                Text("│", style="cyan"),
            ],
            expand=False,
        )

    def _render_metrics_lines(self):
        """渲染指标内容行"""
        if not self.metrics_text:
            return Text("│" + " " * 76 + "│", style="cyan")

        lines = self.metrics_text.split("\n")
        rendered = []

        for line in lines:
            # 每行添加左边框，填充右侧空格，添加右边框
            padded = line[:76].ljust(76)
            rendered.append(Text("│" + padded + "│", style="dim"))

        return Group(*rendered)

    def _format_metrics_text(self, metrics: Dict[str, Any]) -> str:
        """格式化指标为文本

        Args:
            metrics: 指标字典

        Returns:
            格式化的文本字符串
        """
        lines = []

        # 辅助函数：获取指标值，支持多种键名格式
        def get_value(*keys):
            for key in keys:
                if key in metrics:
                    return metrics[key]
            return None

        # 时间信息
        if self.start_time:
            import time
            current_time = time.time()
            elapsed = current_time - self.start_time

            # 计算预计剩余时间
            if self.current_epoch > self.warmup_steps and self.total > 0:
                time_per_epoch = elapsed / max(1, self.current_epoch - self.warmup_steps)
                remaining = time_per_epoch * (self.total - self.current_epoch)
            else:
                remaining = None

            lines.append(f"{'Time elapsed:':<30}{self._format_hms(elapsed)}")
            if remaining is not None:
                lines.append(f"{'ETA:':<30}{self._format_hms(remaining)}")

        # 总步数（支持多种键名格式）
        env_steps = get_value("env_steps", "perf/total_env_steps")
        if env_steps is not None:
            lines.append(f"{'Total steps:':<30}{int(env_steps):,}")

        # SPS (Steps per second) - 支持多种键名格式
        sps = get_value("sps", "perf/steps_per_sec", "perf/avg_steps_per_sec")
        if sps is not None:
            lines.append(f"{'Steps per second:':<30}{int(sps)}")

        # 环境步每秒
        env_sps = get_value("perf/env_steps_per_sec", "perf/avg_env_steps_per_sec")
        if env_sps is not None:
            lines.append(f"{'Env steps per second:':<30}{int(env_sps):,}")

        # Collection/Learning time
        if "collection_time" in metrics:
            lines.append(f"{'Collection time:':<30}{metrics['collection_time']:.3f}s")
        if "learning_time" in metrics:
            lines.append(f"{'Learning time:':<30}{metrics['learning_time']:.3f}s")

        # Iteration time
        iteration_time = get_value("iteration_time", "perf/step_time")
        if iteration_time is not None:
            lines.append(f"{'Iteration time:':<30}{iteration_time:.3f}s")
        elif self.start_time and self.current_epoch > 0:
            import time
            elapsed = time.time() - self.start_time
            lines.append(f"{'Iteration time:':<30}{elapsed / max(1, self.current_epoch):.3f}s")

        lines.append("")  # 空行

        # === 损失指标 ===
        # Value loss
        value_loss = get_value("value_loss", "train/value_loss", "loss/value_loss")
        if value_loss is not None:
            lines.append(f"{'Mean value loss:':<30}{value_loss:.4f}")

        # Policy/Surrogate loss
        policy_loss = get_value("surrogate_loss", "policy_loss", "train/surrogate_loss", "train/policy_loss")
        if policy_loss is not None:
            lines.append(f"{'Mean policy loss:':<30}{policy_loss:.4f}")

        # Entropy loss
        entropy_loss = get_value("entropy_loss", "entropy", "train/entropy_loss", "train/entropy")
        if entropy_loss is not None:
            lines.append(f"{'Mean entropy loss:':<30}{entropy_loss:.4f}")

        # KL divergence
        kl = get_value("approx_kl", "kl", "train/approx_kl")
        if kl is not None:
            lines.append(f"{'Mean KL divergence:':<30}{kl:.4f}")

        # Clip fraction
        clip_frac = get_value("clip_fraction", "train/clip_fraction")
        if clip_frac is not None:
            lines.append(f"{'Clip fraction:':<30}{clip_frac:.2%}")

        lines.append("")  # 空行

        # === 奖励与统计指标 ===
        # Episode reward
        episode_reward = get_value("episode_reward", "train/episode_reward", "reward")
        if episode_reward is not None:
            lines.append(f"{'Mean reward:':<30}{episode_reward:.2f}")

        # Episode length
        episode_length = get_value("episode_length", "train/episode_length")
        if episode_length is not None:
            lines.append(f"{'Mean episode length:':<30}{episode_length:.1f}")

        # 动作噪声
        action_noise = get_value("action_noise_std", "train/action_noise_std")
        if action_noise is not None:
            lines.append(f"{'Mean action noise std:':<30}{action_noise:.3f}")

        # 学习率
        lr = get_value("learning_rate", "train/learning_rate")
        if lr is not None:
            lines.append(f"{'Learning rate:':<30}{lr:.6f}")

        lines.append("")  # 空行

        # === 详细指标（Episode_Reward, Metrics, Curriculum, Episode_Termination） ===
        # 收集所有需要详细显示的指标
        detail_keys = []
        for key in metrics.keys():
            if any(key.startswith(prefix) for prefix in [
                "Episode_Reward/", "Metrics/", "Curriculum/", "Episode_Termination/",
                "train/Episode_Reward/", "train/Metrics/", "train/Curriculum/", "train/Episode_Termination/"
            ]):
                # 跳过已经显示过的核心指标
                if not any(x in key for x in ["episode_reward", "episode_length", "action_noise_std"]):
                    detail_keys.append(key)

        # 排序并显示（限制数量，避免过长）
        for key in sorted(detail_keys)[:20]:  # 最多显示20个详细指标
            value = metrics[key]
            # 格式化键名：移除前缀，替换下划线为空格
            display_key = key
            for prefix in ["train/Episode_Reward/", "train/Metrics/", "train/Curriculum/", "train/Episode_Termination/",
                          "Episode_Reward/", "Metrics/", "Curriculum/", "Episode_Termination/"]:
                if key.startswith(prefix):
                    display_key = key[len(prefix):]
            display_key = display_key.replace("_", " ")
            lines.append(f"{display_key:<30}{float(value):.4f}")

        return "\n".join(lines)

    def _format_hms(self, seconds: float) -> str:
        """格式化时间为 HH:MM:SS

        Args:
            seconds: 秒数

        Returns:
            格式化的时间字符串
        """
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)

        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    def start(self):
        """启动显示"""
        if not self.started:
            self.live.start()
            # 添加进度条任务
            self.progress_task_id = self.progress_renderable.add_task(
                "",
                total=self.total
            )
            self.started = True
            import time
            self.start_time = time.time()

    def update(self, epoch: Optional[int] = None, step: Optional[int] = None, metrics: Optional[Dict[str, Any]] = None):
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
                self.progress_renderable.update(self.progress_task_id, completed=epoch)

        # 更新指标
        if metrics:
            self.current_metrics = metrics.copy()
            self.metrics_text = self._format_metrics_text(metrics)

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
