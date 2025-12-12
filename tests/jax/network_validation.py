"""
神经网络验证脚本
测试Flax网络的前向传播、梯度计算、JIT编译等
使用rich进度条提供清晰反馈
"""

from src.rl.models.optimizer import create_ppo_optimizer
from src.rl.models.ppo import (
    compute_gae_scan, ppo_loss, PPOBatch, prepare_ppo_batch
)
from src.rl.models.networks import (
    ActorNetwork, CriticNetwork, ActorCriticNetwork,
    create_actor_critic, count_parameters
)
from pathlib import Path
import sys
import time
from rich import box
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn
from rich.console import Console
import jax.numpy as jp
import jax
import os
os.environ['JAX_PLATFORMS'] = 'cpu'  # 先用CPU测试


# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


console = Console()


def print_section(title: str):
    """打印分节标题"""
    console.print()
    console.print(Panel(f"[bold cyan]{title}[/bold cyan]", box=box.DOUBLE))


def test_network_creation():
    """测试1: 网络创建"""
    print_section("测试1: 网络创建")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("[cyan]创建网络...", total=4)

        obs_dim = 55  # 机器人观测维度
        action_dim = 14  # 机器人动作维度

        # 创建Actor网络
        actor = ActorNetwork(action_dim=action_dim, hidden_dims=(256, 256))
        progress.update(task, advance=1, description="[cyan]✓ Actor网络创建")

        # 创建Critic网络
        critic = CriticNetwork(hidden_dims=(256, 256))
        progress.update(task, advance=1, description="[cyan]✓ Critic网络创建")

        # 创建共享backbone的Actor-Critic
        ac_shared = ActorCriticNetwork(
            action_dim=action_dim,
            shared_backbone=True,
            hidden_dims=(256, 256)
        )
        progress.update(task, advance=1,
                        description="[cyan]✓ Actor-Critic(共享)创建")

        # 创建分离backbone的Actor-Critic
        ac_separate = ActorCriticNetwork(
            action_dim=action_dim,
            shared_backbone=False,
            hidden_dims=(256, 256)
        )
        progress.update(task, advance=1,
                        description="[cyan]✓ Actor-Critic(分离)创建")

    # 显示网络信息
    info_table = Table(title="网络架构", box=box.ROUNDED)
    info_table.add_column("网络", style="cyan")
    info_table.add_column("类型", style="green")
    info_table.add_column("隐藏层", style="yellow")

    info_table.add_row("Actor", "策略网络", "[256, 256]")
    info_table.add_row("Critic", "价值网络", "[256, 256]")
    info_table.add_row("Actor-Critic(共享)", "联合网络", "[256, 256]")
    info_table.add_row("Actor-Critic(分离)", "联合网络", "[256, 256] × 2")

    console.print(info_table)

    return actor, critic, ac_shared, ac_separate, obs_dim, action_dim


def test_forward_pass(actor, critic, ac_shared, obs_dim, action_dim):
    """测试2: 前向传播"""
    print_section("测试2: 前向传播")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("[cyan]初始化参数...", total=6)

        # 创建假数据
        rng = jax.random.PRNGKey(0)
        batch_size = 32
        obs = jax.random.normal(rng, (batch_size, obs_dim))
        progress.update(task, advance=1, description="[cyan]✓ 创建输入数据")

        # 初始化Actor参数
        rng, key = jax.random.split(rng)
        actor_params = actor.init(key, obs)
        progress.update(task, advance=1, description="[cyan]✓ Actor参数初始化")

        # 初始化Critic参数
        rng, key = jax.random.split(rng)
        critic_params = critic.init(key, obs)
        progress.update(task, advance=1, description="[cyan]✓ Critic参数初始化")

        # 初始化Actor-Critic参数
        rng, key = jax.random.split(rng)
        ac_params = ac_shared.init(key, obs)
        progress.update(task, advance=1,
                        description="[cyan]✓ Actor-Critic参数初始化")

        # 前向传播 - Actor
        mean, log_std = actor.apply(actor_params, obs)
        progress.update(task, advance=1, description="[cyan]✓ Actor前向传播")

        # 前向传播 - Critic
        values = critic.apply(critic_params, obs)
        progress.update(task, advance=1, description="[cyan]✓ Critic前向传播")

    # 显示输出形状
    shape_table = Table(title="输出形状", box=box.ROUNDED)
    shape_table.add_column("网络", style="cyan")
    shape_table.add_column("输出", style="green")
    shape_table.add_column("形状", style="yellow")

    shape_table.add_row("Actor", "mean", str(mean.shape))
    shape_table.add_row("Actor", "log_std", str(log_std.shape))
    shape_table.add_row("Critic", "values", str(values.shape))

    console.print(shape_table)

    # 统计参数数量
    param_table = Table(title="参数统计", box=box.ROUNDED)
    param_table.add_column("网络", style="cyan")
    param_table.add_column("参数数量", style="green", justify="right")

    param_table.add_row("Actor", f"{count_parameters(actor_params):,}")
    param_table.add_row("Critic", f"{count_parameters(critic_params):,}")
    param_table.add_row("Actor-Critic", f"{count_parameters(ac_params):,}")

    console.print(param_table)

    return actor_params, critic_params, ac_params, rng


def test_gradient_computation(ac_shared, ac_params, obs_dim, action_dim):
    """测试3: 梯度计算"""
    print_section("测试3: 梯度计算")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("[cyan]准备数据...", total=4)

        # 创建假批次数据
        rng = jax.random.PRNGKey(42)
        batch_size = 128

        obs = jax.random.normal(rng, (batch_size, obs_dim))
        actions = jax.random.normal(rng, (batch_size, action_dim))
        old_log_probs = jax.random.normal(rng, (batch_size,))
        advantages = jax.random.normal(rng, (batch_size,))
        returns = jax.random.normal(rng, (batch_size,))
        values = jax.random.normal(rng, (batch_size,))

        batch = PPOBatch(
            obs=obs,
            actions=actions,
            old_log_probs=old_log_probs,
            advantages=advantages,
            returns=returns,
            values=values,
        )
        progress.update(task, advance=1, description="[cyan]✓ 批次数据准备")

        # 定义损失函数
        def loss_fn(params):
            loss, info = ppo_loss(params, ac_shared, batch)
            return loss

        progress.update(task, advance=1, description="[cyan]计算损失...")

        # 计算损失
        loss_value = loss_fn(ac_params)
        progress.update(task, advance=1,
                        description=f"[cyan]✓ 损失值: {loss_value:.4f}")

        # 计算梯度
        grads = jax.grad(loss_fn)(ac_params)
        progress.update(task, advance=1, description="[cyan]✓ 梯度计算完成")

    # 统计梯度信息
    grad_norms = jax.tree.map(lambda g: jp.linalg.norm(g), grads)
    flat_norms = jax.tree.leaves(grad_norms)

    grad_table = Table(title="梯度统计", box=box.ROUNDED)
    grad_table.add_column("指标", style="cyan")
    grad_table.add_column("值", style="green", justify="right")

    grad_table.add_row("损失值", f"{float(loss_value):.6f}")
    grad_table.add_row("梯度范数最大值", f"{max(flat_norms):.6f}")
    grad_table.add_row("梯度范数最小值", f"{min(flat_norms):.6f}")
    grad_table.add_row("梯度范数平均值", f"{sum(flat_norms)/len(flat_norms):.6f}")

    console.print(grad_table)

    return grads


def test_jit_compilation(ac_shared, ac_params, obs_dim, action_dim):
    """测试4: JIT编译"""
    print_section("测试4: JIT编译加速")

    # 准备数据
    rng = jax.random.PRNGKey(123)
    obs = jax.random.normal(rng, (32, obs_dim))

    # 非JIT版本
    def forward_no_jit(params, obs):
        return ac_shared.apply(params, obs)

    # JIT版本
    @jax.jit
    def forward_jit(params, obs):
        return ac_shared.apply(params, obs)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        # 非JIT测试
        task1 = progress.add_task("[cyan]非JIT测试 (1000次)...", total=1000)
        start_time = time.time()
        for i in range(1000):
            _ = forward_no_jit(ac_params, obs)
            progress.update(task1, advance=1)
        no_jit_time = time.time() - start_time
        progress.update(
            task1, description=f"[cyan]✓ 非JIT: {no_jit_time*1000:.1f}ms")

        # JIT测试（含编译）
        task2 = progress.add_task("[cyan]JIT测试 (1000次, 含编译)...", total=1000)
        start_time = time.time()
        for i in range(1000):
            _ = forward_jit(ac_params, obs)
            jax.block_until_ready(_[0])
            progress.update(task2, advance=1)
        jit_time_with_compile = time.time() - start_time
        progress.update(
            task2, description=f"[cyan]✓ JIT(含编译): {jit_time_with_compile*1000:.1f}ms")

        # JIT测试（纯执行，预热后）
        task3 = progress.add_task("[cyan]JIT测试 (1000次, 纯执行)...", total=1000)
        # 预热
        for _ in range(10):
            _ = forward_jit(ac_params, obs)
            jax.block_until_ready(_[0])

        start_time = time.time()
        for i in range(1000):
            _ = forward_jit(ac_params, obs)
            jax.block_until_ready(_[0])
            progress.update(task3, advance=1)
        jit_time = time.time() - start_time
        progress.update(
            task3, description=f"[cyan]✓ JIT(纯执行): {jit_time*1000:.1f}ms")

    # 计算加速比
    speedup = no_jit_time / jit_time if jit_time > 0 else 0

    perf_table = Table(title="JIT性能对比", box=box.ROUNDED)
    perf_table.add_column("方式", style="cyan")
    perf_table.add_column("总时间 (ms)", style="green", justify="right")
    perf_table.add_column("加速比", style="yellow", justify="right")

    perf_table.add_row("非JIT", f"{no_jit_time*1000:.1f}", "1.0x")
    perf_table.add_row("JIT(含编译)", f"{jit_time_with_compile*1000:.1f}",
                       f"{no_jit_time/jit_time_with_compile:.1f}x")
    perf_table.add_row("JIT(纯执行)", f"{jit_time*1000:.1f}", f"{speedup:.1f}x")

    console.print(perf_table)


def test_gae_computation():
    """测试5: GAE计算"""
    print_section("测试5: GAE广义优势估计")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("[cyan]测试GAE...", total=3)

        # 创建假轨迹数据
        T = 100
        rewards = jp.ones(T) * 0.1
        values = jp.ones(T + 1) * 0.5
        dones = jp.zeros(T)
        progress.update(task, advance=1, description="[cyan]✓ 创建轨迹数据")

        # 计算GAE
        advantages, returns = compute_gae_scan(
            rewards=rewards,
            values=values,
            dones=dones,
            gamma=0.99,
            gae_lambda=0.95,
        )
        progress.update(task, advance=1, description="[cyan]✓ GAE计算完成")

        # 验证形状
        assert advantages.shape == (T,)
        assert returns.shape == (T,)
        progress.update(task, advance=1, description="[cyan]✓ 形状验证通过")

    # 显示GAE统计
    gae_table = Table(title="GAE统计", box=box.ROUNDED)
    gae_table.add_column("指标", style="cyan")
    gae_table.add_column("值", style="green", justify="right")

    gae_table.add_row("轨迹长度", str(T))
    gae_table.add_row("优势均值", f"{float(advantages.mean()):.6f}")
    gae_table.add_row("优势标准差", f"{float(advantages.std()):.6f}")
    gae_table.add_row("回报均值", f"{float(returns.mean()):.6f}")
    gae_table.add_row("回报标准差", f"{float(returns.std()):.6f}")

    console.print(gae_table)


def main():
    """主测试流程"""
    console.print(Panel.fit(
        "[bold green]Flax网络与PPO算法验证[/bold green]\n"
        "[dim]测试网络创建、前向传播、梯度计算、JIT编译、GAE计算[/dim]",
        border_style="green"
    ))

    try:
        # 测试1: 网络创建
        actor, critic, ac_shared, ac_separate, obs_dim, action_dim = test_network_creation()

        # 测试2: 前向传播
        actor_params, critic_params, ac_params, rng = test_forward_pass(
            actor, critic, ac_shared, obs_dim, action_dim
        )

        # 测试3: 梯度计算
        grads = test_gradient_computation(
            ac_shared, ac_params, obs_dim, action_dim)

        # 测试4: JIT编译
        test_jit_compilation(ac_shared, ac_params, obs_dim, action_dim)

        # 测试5: GAE计算
        test_gae_computation()

        # 成功总结
        console.print()
        console.print(Panel.fit(
            "[bold green]✓ 所有测试通过！[/bold green]\n"
            "[dim]网络已准备好用于PPO训练[/dim]",
            border_style="green"
        ))

    except Exception as e:
        console.print()
        console.print(Panel.fit(
            f"[bold red]✗ 测试失败[/bold red]\n"
            f"[red]{e}[/red]",
            border_style="red"
        ))
        import traceback
        console.print(traceback.format_exc())
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
