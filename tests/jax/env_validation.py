"""
MJX环境验证脚本
测试机器人环境的reset/step/JIT/vmap功能
使用rich进度条提供清晰的反馈
"""

from src.jiyuan_rl.envs.jiyuan_mjx_env import JiyuanMJXEnv
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


def test_basic_creation():
    """测试1: 环境创建"""
    print_section("测试1: 环境创建")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("[cyan]创建环境...", total=3)

        # 尝试使用Open_Duck_Playground的模型
        try:
            env = JiyuanMJXEnv(
                xml_path="/home/chenpeel/Desktop/duck/Open_Duck_Playground/playground/open_duck_mini_v2/xmls/scene_flat_terrain.xml",
                verbose=False,
            )
            progress.update(task, advance=1, description="[cyan]✓ 环境创建成功")
        except Exception as e:
            console.print(
                f"[yellow]⚠ 无法加载Open_Duck_Playground模型: {e}[/yellow]")
            console.print("[yellow]将使用简单的测试模型...[/yellow]")

            # 创建一个简单的测试XML
            test_xml = create_simple_test_model()
            progress.update(task, advance=1, description="[cyan]创建简单测试模型")

            env = JiyuanMJXEnv(xml_path=test_xml, verbose=False)
            progress.update(task, advance=1, description="[cyan]✓ 测试环境创建成功")

        progress.update(task, completed=3)

    # 显示环境信息
    info_table = Table(title="环境信息", box=box.ROUNDED)
    info_table.add_column("属性", style="cyan")
    info_table.add_column("值", style="green")

    info_table.add_row("观测维度", str(env.observation_size))
    info_table.add_row("动作维度", str(env.action_size))
    info_table.add_row("最大步数", str(env.max_steps))
    info_table.add_row("控制频率", f"{env.control_freq:.1f} Hz")
    info_table.add_row("仿真时间步", f"{env.dt:.4f} s")

    console.print(info_table)

    return env


def test_reset_and_step(env: JiyuanMJXEnv):
    """测试2: Reset和Step功能"""
    print_section("测试2: Reset和Step功能")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("[cyan]测试环境操作...", total=4)

        # 创建随机数生成器
        rng = jax.random.PRNGKey(0)
        progress.update(task, advance=1, description="[cyan]初始化RNG")

        # 测试reset
        state = env.reset(rng)
        progress.update(task, advance=1, description="[cyan]✓ Reset成功")

        # 测试step
        action = jp.zeros(env.action_size)
        state = env.step(state, action)
        progress.update(task, advance=1, description="[cyan]✓ Step成功")

        # 运行几步
        for i in range(10):
            action = jax.random.uniform(
                rng, (env.action_size,), minval=-0.1, maxval=0.1)
            state = env.step(state, action)
            rng, _ = jax.random.split(rng)

        progress.update(task, advance=1, description="[cyan]✓ 多步运行成功")

    # 显示状态信息
    state_table = Table(title="环境状态", box=box.ROUNDED)
    state_table.add_column("属性", style="cyan")
    state_table.add_column("值", style="green")

    state_table.add_row("当前步数", str(int(state.step)))
    state_table.add_row("累积奖励", f"{float(state.reward):.4f}")
    state_table.add_row("是否终止", str(bool(state.done)))
    state_table.add_row("观测形状", str(state.obs.shape))

    console.print(state_table)

    return state


def test_jit_compilation(env: JiyuanMJXEnv):
    """测试3: JIT编译加速"""
    print_section("测试3: JIT编译加速")

    # 编译函数
    @jax.jit
    def jit_reset(rng):
        return env.reset(rng)

    @jax.jit
    def jit_step(state, action):
        return env.step(state, action)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        # 1. 非JIT性能测试
        task1 = progress.add_task("[cyan]非JIT性能测试 (100次reset)...", total=100)

        rng = jax.random.PRNGKey(42)
        start_time = time.time()

        for i in range(100):
            rng, reset_rng = jax.random.split(rng)
            state = env.reset(reset_rng)
            progress.update(task1, advance=1)

        no_jit_time = time.time() - start_time
        progress.update(
            task1, description=f"[cyan]✓ 非JIT: {no_jit_time*1000:.1f}ms")

        # 2. JIT性能测试（含编译时间）
        task2 = progress.add_task(
            "[cyan]JIT性能测试 (100次reset, 含编译)...", total=100)

        rng = jax.random.PRNGKey(42)
        start_time = time.time()

        for i in range(100):
            rng, reset_rng = jax.random.split(rng)
            state = jit_reset(reset_rng)
            # 等待JAX完成计算
            jax.block_until_ready(state.obs)
            progress.update(task2, advance=1)

        jit_time_with_compile = time.time() - start_time
        progress.update(
            task2, description=f"[cyan]✓ JIT(含编译): {jit_time_with_compile*1000:.1f}ms")

        # 3. JIT性能测试（纯执行）
        task3 = progress.add_task(
            "[cyan]JIT性能测试 (100次reset, 纯执行)...", total=100)

        rng = jax.random.PRNGKey(42)
        # 预热
        for i in range(5):
            rng, reset_rng = jax.random.split(rng)
            state = jit_reset(reset_rng)
            jax.block_until_ready(state.obs)

        rng = jax.random.PRNGKey(42)
        start_time = time.time()

        for i in range(100):
            rng, reset_rng = jax.random.split(rng)
            state = jit_reset(reset_rng)
            jax.block_until_ready(state.obs)
            progress.update(task3, advance=1)

        jit_time = time.time() - start_time
        progress.update(
            task3, description=f"[cyan]✓ JIT(纯执行): {jit_time*1000:.1f}ms")

    # 计算加速比
    speedup = no_jit_time / jit_time if jit_time > 0 else 0

    perf_table = Table(title="JIT性能对比", box=box.ROUNDED)
    perf_table.add_column("方式", style="cyan")
    perf_table.add_column("总时间 (ms)", style="green", justify="right")
    perf_table.add_column("平均时间 (ms)", style="green", justify="right")
    perf_table.add_column("加速比", style="yellow", justify="right")

    perf_table.add_row(
        "非JIT", f"{no_jit_time*1000:.1f}", f"{no_jit_time*10:.2f}", "1.0x")
    perf_table.add_row("JIT(含编译)", f"{jit_time_with_compile*1000:.1f}",
                       f"{jit_time_with_compile*10:.2f}", f"{no_jit_time/jit_time_with_compile:.1f}x")
    perf_table.add_row(
        "JIT(纯执行)", f"{jit_time*1000:.1f}", f"{jit_time*10:.2f}", f"{speedup:.1f}x")

    console.print(perf_table)


def test_vmap_batch(env: JiyuanMJXEnv):
    """测试4: vmap批量并行"""
    print_section("测试4: vmap批量并行")

    batch_sizes = [4, 16, 64]

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:

        for batch_size in batch_sizes:
            task = progress.add_task(
                f"[cyan]测试批量大小 {batch_size}...",
                total=3
            )

            # 批量reset
            rng = jax.random.PRNGKey(0)
            start_time = time.time()

            states = env.batch_reset(rng, batch_size)
            reset_time = time.time() - start_time

            progress.update(
                task, advance=1, description=f"[cyan]Reset {batch_size}个环境: {reset_time*1000:.1f}ms")

            # 批量step
            actions = jax.random.uniform(
                rng, (batch_size, env.action_size), minval=-0.1, maxval=0.1)
            start_time = time.time()

            states = env.batch_step(states, actions)
            step_time = time.time() - start_time

            progress.update(
                task, advance=1, description=f"[cyan]Step {batch_size}个环境: {step_time*1000:.1f}ms")

            # 验证形状
            assert states.obs.shape == (batch_size, env.observation_size)
            assert states.reward.shape == (batch_size,)
            assert states.done.shape == (batch_size,)

            progress.update(
                task, advance=1, description=f"[green]✓ 批量{batch_size}: Reset={reset_time*1000:.1f}ms, Step={step_time*1000:.1f}ms")

    console.print("[green]✓ 所有批量测试通过[/green]")


def test_full_episode(env: JiyuanMJXEnv):
    """测试5: 完整episode运行"""
    print_section("测试5: 完整Episode运行")

    # JIT编译
    @jax.jit
    def jit_reset(rng):
        return env.reset(rng)

    @jax.jit
    def jit_step(state, action):
        return env.step(state, action)

    num_steps = 100

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("[cyan]运行episode...", total=num_steps)

        # Reset
        rng = jax.random.PRNGKey(123)
        state = jit_reset(rng)
        jax.block_until_ready(state.obs)
        progress.update(task, advance=0, description="[cyan]✓ Reset完成")

        # 运行episode
        total_reward = 0.0
        start_time = time.time()

        for i in range(num_steps):
            # 随机动作
            rng, action_rng = jax.random.split(rng)
            action = jax.random.uniform(
                action_rng, (env.action_size,), minval=-0.2, maxval=0.2)

            # Step
            state = jit_step(state, action)
            jax.block_until_ready(state.obs)
            total_reward += float(state.reward)

            progress.update(
                task,
                advance=1,
                description=f"[cyan]Step {i+1}/{num_steps} | Reward: {total_reward:.2f}"
            )

            if state.done:
                progress.update(task, completed=num_steps,
                                description=f"[yellow]Episode提前终止于步数{i+1}")
                break

        episode_time = time.time() - start_time

        if not state.done:
            progress.update(task, description=f"[green]✓ Episode完成")

    # 显示episode统计
    stats_table = Table(title="Episode统计", box=box.ROUNDED)
    stats_table.add_column("指标", style="cyan")
    stats_table.add_column("值", style="green")

    stats_table.add_row("总步数", str(int(state.step)))
    stats_table.add_row("总奖励", f"{total_reward:.4f}")
    stats_table.add_row("平均奖励", f"{total_reward/int(state.step):.4f}")
    stats_table.add_row("总时间", f"{episode_time*1000:.1f} ms")
    stats_table.add_row("每步时间", f"{episode_time*1000/num_steps:.2f} ms")
    stats_table.add_row("SPS (steps/sec)", f"{num_steps/episode_time:.1f}")

    console.print(stats_table)


def create_simple_test_model() -> str:
    """创建一个简单的测试模型（如果无法访问Open_Duck_Playground）"""
    xml_content = """
    <mujoco model="simple_test">
        <option timestep="0.002"/>
        <worldbody>
            <body name="torso" pos="0 0 0.5">
                <freejoint/>
                <geom type="box" size="0.1 0.1 0.1" mass="1"/>
                <body name="leg" pos="0 0 -0.2">
                    <joint name="hip" type="hinge" axis="0 1 0"/>
                    <geom type="capsule" size="0.02" fromto="0 0 0 0 0 -0.3" mass="0.5"/>
                </body>
            </body>
        </worldbody>
        <actuator>
            <motor joint="hip" gear="100"/>
        </actuator>
    </mujoco>
    """

    # 保存到临时文件
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False) as f:
        f.write(xml_content)
        return f.name


def main():
    """主测试流程"""
    console.print(Panel.fit(
        "[bold green]机器人MJX环境验证[/bold green]\n"
        "[dim]测试环境创建、reset/step、JIT编译、vmap批量、完整episode[/dim]",
        border_style="green"
    ))

    try:
        # 测试1: 创建环境
        env = test_basic_creation()

        # 测试2: Reset和Step
        state = test_reset_and_step(env)

        # 测试3: JIT编译
        test_jit_compilation(env)

        # 测试4: vmap批量
        test_vmap_batch(env)

        # 测试5: 完整episode
        test_full_episode(env)

        # 成功总结
        console.print()
        console.print(Panel.fit(
            "[bold green]✓ 所有测试通过！[/bold green]\n"
            "[dim]环境已准备好用于训练[/dim]",
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
