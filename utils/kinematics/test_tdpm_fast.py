"""
Cython优化版tdpm的性能测试与对比
运行此脚本以验证：
1. 编译成功
2. 功能正确性
3. 性能提升
"""

import numpy as np
import time
from rich.console import Console
from rich.table import Table
from rich.progress import track

# 导入原始Python版本
from tdpm import ThreeDOFParallelMechanism

# 导入Cython优化版本
try:
    from tdpm_fast import ThreeDOFParallelMechanismFast
    CYTHON_AVAILABLE = True
except ImportError:
    CYTHON_AVAILABLE = False
    print("❌ Cython版本未编译，请先运行：")
    print("   cd utils/kinematics && python setup_tdpm.py build_ext --inplace")

console = Console()

def test_correctness():
    """测试Cython版本与Python版本的数值一致性"""
    console.print("\n[bold cyan]1. 正确性测试[/bold cyan]")

    if not CYTHON_AVAILABLE:
        return

    # 初始化两个版本
    l0, l1, l2 = 0.02, 0.02, 0.02
    py_tdpm = ThreeDOFParallelMechanism(l0, l1, l2)
    cy_tdpm = ThreeDOFParallelMechanismFast(l0, l1, l2)

    # 测试用例
    test_cases = [
        np.array([0.0, 0.0, 0.0]),
        np.array([0.1, 0.1, 0.1]),
        np.array([-0.2, 0.15, -0.1]),
        np.array([0.5, -0.3, 0.4]),  # 接近边界
    ]

    table = Table(title="正向运动学对比")
    table.add_column("测试用例", style="cyan")
    table.add_column("Python theta_a", style="green")
    table.add_column("Cython theta_a", style="green")
    table.add_column("误差", style="yellow")
    table.add_column("状态", style="magenta")

    max_error = 0.0
    for i, mu in enumerate(test_cases):
        # Python版本
        py_result = py_tdpm.forward_kinematics(mu)

        # Cython版本
        cy_result = cy_tdpm.forward_kinematics(mu)

        # 比较theta_a
        py_theta_a = py_result['theta_a']
        cy_theta_a = cy_result['theta_a']
        error = abs(py_theta_a - cy_theta_a)
        max_error = max(max_error, error)

        status = "✅ PASS" if error < 1e-10 else "❌ FAIL"

        table.add_row(
            f"mu_{i}",
            f"{py_theta_a:.10f}",
            f"{cy_theta_a:.10f}",
            f"{error:.2e}",
            status
        )

    console.print(table)
    console.print(f"\n最大误差: {max_error:.2e} (阈值: 1e-10)")

    if max_error < 1e-10:
        console.print("[bold green]✅ 正确性测试通过！[/bold green]")
    else:
        console.print("[bold red]❌ 正确性测试失败！[/bold red]")

    return max_error < 1e-10


def test_safety_smoothing():
    """测试二阶平滑限制功能"""
    console.print("\n[bold cyan]2. 二阶平滑限制测试[/bold cyan]")

    if not CYTHON_AVAILABLE:
        return

    l0, l1, l2 = 0.02, 0.02, 0.02
    cy_tdpm = ThreeDOFParallelMechanismFast(
        l0, l1, l2,
        dt=0.01,                    # 100Hz控制
        max_angle=np.pi/6,          # ±30°
        max_velocity=2.0,           # 2 rad/s
        max_acceleration=10.0       # 10 rad/s²
    )

    # 模拟突变指令（从0突变到最大角度）
    console.print("\n[yellow]模拟场景：原始指令从0突变到π/6[/yellow]")

    trajectory = []
    velocities = []
    accelerations = []

    raw_commands = [
        np.array([0.0, 0.0, 0.0]),  # t=0: 初始
        np.array([np.pi/6, 0.0, 0.0]),  # t=1: 突变到最大值
    ]

    # 模拟100步 (1秒)
    for step in range(100):
        if step == 0:
            raw_mu = raw_commands[0]
        else:
            raw_mu = raw_commands[1]  # 保持最大值指令

        safe_mu = cy_tdpm.safe_step(raw_mu)
        trajectory.append(safe_mu[0])  # 只记录roll轴

        if step > 0:
            vel = (trajectory[-1] - trajectory[-2]) / 0.01
            velocities.append(vel)

            if step > 1:
                acc = (velocities[-1] - velocities[-2]) / 0.01
                accelerations.append(acc)

    # 检查约束
    max_vel = max(abs(v) for v in velocities)
    max_acc = max(abs(a) for a in accelerations)
    final_angle = trajectory[-1]

    table = Table(title="平滑限制效果")
    table.add_column("指标", style="cyan")
    table.add_column("测量值", style="green")
    table.add_column("限制值", style="yellow")
    table.add_column("状态", style="magenta")

    table.add_row(
        "最大角速度",
        f"{max_vel:.3f} rad/s",
        "2.0 rad/s",
        "✅" if max_vel <= 2.0 else "❌"
    )
    table.add_row(
        "最大角加速度",
        f"{max_acc:.3f} rad/s²",
        "10.0 rad/s²",
        "✅" if max_acc <= 10.0 else "❌"
    )
    table.add_row(
        "最终角度",
        f"{final_angle:.3f} rad",
        f"{np.pi/6:.3f} rad",
        "✅" if abs(final_angle - np.pi/6) < 0.01 else "❌"
    )

    console.print(table)

    # 绘制简单的ASCII轨迹图
    console.print("\n[cyan]角度变化轨迹（前20步）：[/cyan]")
    for i in range(min(20, len(trajectory))):
        bar_length = int(trajectory[i] / (np.pi/6) * 40)
        bar = "█" * bar_length
        console.print(f"Step {i:2d}: {bar} {trajectory[i]:.4f}")

    if max_vel <= 2.0 and max_acc <= 10.0:
        console.print("\n[bold green]✅ 平滑限制测试通过！[/bold green]")
        console.print("[dim]说明：即使指令突变，实际输出也会平滑过渡[/dim]")
    else:
        console.print("\n[bold red]❌ 平滑限制测试失败！[/bold red]")


def benchmark_performance():
    """性能基准测试"""
    console.print("\n[bold cyan]3. 性能基准测试[/bold cyan]")

    if not CYTHON_AVAILABLE:
        return

    l0, l1, l2 = 0.02, 0.02, 0.02
    py_tdpm = ThreeDOFParallelMechanism(l0, l1, l2)
    cy_tdpm = ThreeDOFParallelMechanismFast(l0, l1, l2)

    # 准备测试数据
    n_iterations = 10000
    test_mu = np.random.uniform(-0.3, 0.3, (n_iterations, 3))

    console.print(f"\n[yellow]测试规模：{n_iterations:,} 次正向运动学计算[/yellow]")

    # Python版本
    console.print("\n[dim]测试Python版本...[/dim]")
    start = time.perf_counter()
    for mu in track(test_mu[:1000], description="Python (1000次)"):  # 只测试1000次避免太慢
        _ = py_tdpm.forward_kinematics(mu)
    py_time_1k = time.perf_counter() - start
    py_time = py_time_1k * 10  # 估算10000次的时间

    # Cython版本
    console.print("\n[dim]测试Cython版本...[/dim]")
    start = time.perf_counter()
    for mu in track(test_mu, description="Cython (10000次)"):
        _ = cy_tdpm.forward_kinematics(mu)
    cy_time = time.perf_counter() - start

    # 性能对比
    speedup = py_time / cy_time

    table = Table(title="性能对比")
    table.add_column("版本", style="cyan")
    table.add_column("总耗时", style="green")
    table.add_column("单次耗时", style="yellow")
    table.add_column("吞吐量", style="magenta")

    table.add_row(
        "Python",
        f"{py_time:.3f} s",
        f"{py_time/n_iterations*1e6:.2f} μs",
        f"{n_iterations/py_time:.0f} ops/s"
    )
    table.add_row(
        "Cython",
        f"{cy_time:.3f} s",
        f"{cy_time/n_iterations*1e6:.2f} μs",
        f"{n_iterations/cy_time:.0f} ops/s"
    )
    table.add_row(
        "[bold]加速比[/bold]",
        f"[bold green]{speedup:.1f}x[/bold green]",
        "",
        ""
    )

    console.print(table)

    # 测试safe_step性能（Cython独有）
    console.print("\n[dim]测试二阶平滑限制性能（safe_step）...[/dim]")
    cy_tdpm.reset_state()
    start = time.perf_counter()
    for mu in track(test_mu, description="safe_step (10000次)"):
        _ = cy_tdpm.safe_step(mu)
    safe_step_time = time.perf_counter() - start

    console.print(f"\n[cyan]safe_step性能：[/cyan]")
    console.print(f"  总耗时: {safe_step_time:.3f} s")
    console.print(f"  单次耗时: {safe_step_time/n_iterations*1e6:.2f} μs")
    console.print(f"  吞吐量: {n_iterations/safe_step_time:.0f} ops/s")

    # 评估是否满足实时性要求
    single_call_us = cy_time / n_iterations * 1e6
    control_freq_max = 1e6 / single_call_us

    console.print(f"\n[bold cyan]实时性评估：[/bold cyan]")
    console.print(f"  单次调用耗时: {single_call_us:.2f} μs")
    console.print(f"  理论最大控制频率: {control_freq_max/1000:.1f} kHz")

    if control_freq_max >= 1000:  # 1kHz
        console.print("[bold green]✅ 满足1kHz实时控制要求！[/bold green]")
    else:
        console.print("[bold yellow]⚠️  可能无法满足高频实时控制[/bold yellow]")


def test_safety_fallback():
    """测试安全回退机制"""
    console.print("\n[bold cyan]4. 安全回退测试[/bold cyan]")

    if not CYTHON_AVAILABLE:
        return

    l0, l1, l2 = 0.02, 0.02, 0.02
    cy_tdpm = ThreeDOFParallelMechanismFast(l0, l1, l2)

    console.print("[yellow]模拟场景：输入导致逆运动学奇异的姿态[/yellow]")

    # 先输入一个有效姿态
    valid_mu = np.array([0.1, 0.1, 0.1])
    safe_mu_1 = cy_tdpm.safe_step(valid_mu)
    console.print(f"第1步（有效）: {safe_mu_1}")

    # 然后输入一个可能导致奇异的姿态（过大角度）
    singular_mu = np.array([2.0, 2.0, 2.0])  # 远超过π/6限制
    safe_mu_2 = cy_tdpm.safe_step(singular_mu)
    console.print(f"第2步（奇异输入）: {safe_mu_2}")

    # 检查是否成功限制
    if np.max(np.abs(safe_mu_2)) <= np.pi/6:
        console.print("[bold green]✅ 成功限制在安全范围内！[/bold green]")
    else:
        console.print("[bold red]❌ 安全限制失败！[/bold red]")

    # 显示安全性指标
    metrics = cy_tdpm.get_safety_metrics()
    console.print("\n[cyan]安全性指标：[/cyan]")
    console.print(f"  当前角度: {metrics['current_angle']}")
    console.print(f"  当前速度: {metrics['current_velocity']}")
    console.print(f"  角度余量: {metrics['angle_margin']:.4f} rad")
    console.print(f"  速度余量: {metrics['velocity_margin']:.4f} rad/s")
    console.print(f"  安全状态: {'✅' if metrics['is_safe'] else '❌'}")


if __name__ == "__main__":
    console.print("[bold magenta]Cython优化版TDPM测试套件[/bold magenta]")
    console.print("[dim]作者: jiyuan_rl | 版本: 1.0[/dim]")

    if not CYTHON_AVAILABLE:
        console.print("\n[bold red]请先编译Cython扩展：[/bold red]")
        console.print("[yellow]cd utils/kinematics[/yellow]")
        console.print("[yellow]python setup_tdpm.py build_ext --inplace[/yellow]")
        exit(1)

    # 运行所有测试
    test_correctness()
    test_safety_smoothing()
    benchmark_performance()
    test_safety_fallback()

    console.print("\n[bold green]🎉 所有测试完成！[/bold green]")
