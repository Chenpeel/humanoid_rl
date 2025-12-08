"""
快速MJX环境验证脚本（GPU版本）
专注于核心功能测试，减少重复次数
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
os.environ['JAX_PLATFORMS'] = 'cuda'  # 使用GPU


# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


console = Console()


def print_device_info():
    """打印设备信息"""
    devices = jax.devices()
    table = Table(title="JAX设备信息", box=box.ROUNDED)
    table.add_column("索引", style="cyan")
    table.add_column("设备类型", style="green")
    table.add_column("设备名称", style="yellow")

    for i, device in enumerate(devices):
        table.add_row(str(i), str(device.platform), str(device))

    console.print(table)


def main():
    """主测试流程"""
    console.print(Panel.fit(
        "[bold green]机器人MJX环境验证 (GPU)[/bold green]\n"
        "[dim]快速版本：测试核心功能 + GPU加速[/dim]",
        border_style="green"
    ))

    # 显示设备信息
    print_device_info()

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=console,
        ) as progress:

            # 1. 创建环境
            task1 = progress.add_task("[cyan]创建环境...", total=1)
            env = JiyuanMJXEnv(
                xml_path="/home/chenpeel/Desktop/duck/Open_Duck_Playground/playground/open_duck_mini_v2/xmls/scene_flat_terrain.xml",
                verbose=False,
            )
            progress.update(task1, completed=1, description="[green]✓ 环境创建成功")

            # 2. 测试Reset
            task2 = progress.add_task("[cyan]测试Reset...", total=1)
            rng = jax.random.PRNGKey(0)
            state = env.reset(rng)
            progress.update(task2, completed=1, description="[green]✓ Reset成功")

            # 3. 测试Step
            task3 = progress.add_task("[cyan]测试Step...", total=10)
            for i in range(10):
                action = jp.zeros(env.action_size)
                state = env.step(state, action)
                progress.update(task3, advance=1)
            progress.update(task3, description="[green]✓ Step成功 (10步)")

            # 4. JIT编译测试
            task4 = progress.add_task("[cyan]JIT编译测试...", total=3)

            @jax.jit
            def jit_reset(rng):
                return env.reset(rng)

            @jax.jit
            def jit_step(state, action):
                return env.step(state, action)

            # 编译（第一次调用）
            rng = jax.random.PRNGKey(42)
            state = jit_reset(rng)
            jax.block_until_ready(state.obs)
            progress.update(task4, advance=1, description="[cyan]JIT编译完成...")

            # 性能测试
            rng = jax.random.PRNGKey(42)
            start_time = time.time()
            for i in range(20):
                rng, reset_rng = jax.random.split(rng)
                state = jit_reset(reset_rng)
                jax.block_until_ready(state.obs)
            jit_time = time.time() - start_time
            progress.update(
                task4, advance=1, description=f"[cyan]JIT性能测试: {jit_time*1000:.1f}ms (20次)")

            # Step性能测试
            state = jit_reset(rng)
            action = jp.zeros(env.action_size)
            start_time = time.time()
            for i in range(100):
                state = jit_step(state, action)
                jax.block_until_ready(state.obs)
            step_time = time.time() - start_time
            progress.update(
                task4, advance=1, description=f"[green]✓ JIT Step: {step_time*1000:.1f}ms (100步)")

            # 5. vmap批量测试
            task5 = progress.add_task("[cyan]vmap批量测试...", total=3)

            batch_sizes = [4, 16, 64]
            batch_results = []

            for batch_size in batch_sizes:
                rng = jax.random.PRNGKey(0)
                start_time = time.time()

                # 批量reset
                states = env.batch_reset(rng, batch_size)
                jax.block_until_ready(states.obs)
                reset_time = time.time() - start_time

                # 批量step
                actions = jp.zeros((batch_size, env.action_size))
                start_time = time.time()
                states = env.batch_step(states, actions)
                jax.block_until_ready(states.obs)
                step_time = time.time() - start_time

                batch_results.append((batch_size, reset_time, step_time))
                progress.update(
                    task5, advance=1, description=f"[cyan]批量{batch_size}: Reset={reset_time*1000:.1f}ms, Step={step_time*1000:.1f}ms")

            progress.update(task5, description="[green]✓ vmap批量测试完成")

        # 显示结果表格
        console.print()

        # 环境信息
        info_table = Table(title="环境信息", box=box.ROUNDED)
        info_table.add_column("属性", style="cyan")
        info_table.add_column("值", style="green")

        info_table.add_row("观测维度", str(env.observation_size))
        info_table.add_row("动作维度", str(env.action_size))
        info_table.add_row("控制频率", f"{env.control_freq:.1f} Hz")
        info_table.add_row("当前步数", str(int(state.step)))

        console.print(info_table)

        # 性能结果
        perf_table = Table(title="性能测试结果 (GPU)", box=box.ROUNDED)
        perf_table.add_column("测试项", style="cyan")
        perf_table.add_column("时间", style="green", justify="right")
        perf_table.add_column("SPS", style="yellow", justify="right")

        perf_table.add_row(
            "JIT Reset (20次)",
            f"{jit_time*1000:.1f} ms",
            f"{20/jit_time:.0f} resets/s"
        )
        perf_table.add_row(
            "JIT Step (100次)",
            f"{step_time*1000:.1f} ms",
            f"{100/step_time:.0f} steps/s"
        )

        for batch_size, reset_time, step_time in batch_results:
            perf_table.add_row(
                f"Batch-{batch_size} Reset",
                f"{reset_time*1000:.2f} ms",
                f"{batch_size/reset_time:.0f} envs/s"
            )
            perf_table.add_row(
                f"Batch-{batch_size} Step",
                f"{step_time*1000:.2f} ms",
                f"{batch_size/step_time:.0f} envs/s"
            )

        console.print(perf_table)

        # 成功总结
        console.print()
        console.print(Panel.fit(
            "[bold green]✓ 所有测试通过！[/bold green]\n"
            f"[dim]GPU: {jax.devices()[0]}[/dim]\n"
            f"[dim]观测维度: {env.observation_size} | 动作维度: {env.action_size}[/dim]\n"
            f"[dim]JIT性能: {100/step_time:.0f} steps/s | 批量64: {64/batch_results[2][1]:.0f} envs/s[/dim]",
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
