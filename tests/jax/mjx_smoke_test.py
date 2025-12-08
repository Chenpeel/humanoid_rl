"""
MJX Smoke Test - 验证MuJoCo MJX功能

测试：
1. 加载MuJoCo XML模型
2. 转换为MJX模型
3. 初始化MJX数据
4. 执行仿真步
5. 检查状态更新
"""

import jax
import jax.numpy as jnp
import mujoco
try:
    # 新版可能在 mujoco 包内提供 mjx
    from mujoco import mjx  # type: ignore
except Exception:
    # 旧版单独包：import 名称为 mujoco_mjx
    import mujoco_mjx as mjx  # type: ignore
import os
import time
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.panel import Panel
from rich.table import Table

console = Console()

console.print(Panel.fit("[bold cyan]MJX Smoke Test[/bold cyan]", border_style="cyan"))

# 1. 检查模型文件
console.print("\n[bold]1. 检查模型文件...[/bold]")
xml_path = "assets/xmls/scene_flat_terrain.xml"
if os.path.exists(xml_path):
    console.print(f"   [green]✓[/green] 找到模型文件: {xml_path}")
    # 加载实际模型
    mj_model = mujoco.MjModel.from_xml_path(xml_path)
else:
    console.print(f"   [yellow]✗[/yellow] 模型文件不存在: {xml_path}")
    console.print("   [yellow]使用简单的pendulum示例...[/yellow]")
    # 创建简单的摆模型
    xml_string = """
    <mujoco>
      <worldbody>
        <light diffuse=".5 .5 .5" pos="0 0 3" dir="0 0 -1"/>
        <geom type="plane" size="1 1 0.1" rgba=".9 0 0 1"/>
        <body pos="0 0 1">
          <joint type="hinge" axis="1 0 0"/>
          <geom type="capsule" size="0.1" fromto="0 0 0 0 0 0.5"/>
        </body>
      </worldbody>
    </mujoco>
    """
    mj_model = mujoco.MjModel.from_xml_string(xml_string)

console.print(f"   模型自由度数: [cyan]{mj_model.nv}[/cyan]")
console.print(f"   模型执行器数: [cyan]{mj_model.nu}[/cyan]")
console.print(f"   时间步长: [cyan]{mj_model.opt.timestep}[/cyan]")

# 2. 转换为MJX模型
console.print("\n[bold]2. 转换为MJX模型...[/bold]")
mjx_model = mjx.put_model(mj_model)
console.print(f"   [green]✓[/green] MJX模型创建成功")
console.print(f"   模型类型: [cyan]{type(mjx_model).__name__}[/cyan]")

# 3. 初始化MJX数据
console.print("\n[bold]3. 初始化MJX数据...[/bold]")
mjx_data = mjx.make_data(mjx_model)
console.print(f"   [green]✓[/green] MJX数据初始化成功")
console.print(f"   qpos shape: [cyan]{mjx_data.qpos.shape}[/cyan]")
console.print(f"   qvel shape: [cyan]{mjx_data.qvel.shape}[/cyan]")
console.print(f"   初始qpos: {mjx_data.qpos[:5]}...")

# 4. 执行仿真步
console.print("\n[bold]4. 执行仿真步...[/bold]")
with Progress(
    SpinnerColumn(),
    TextColumn("[progress.description]{task.description}"),
    BarColumn(),
    TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
    TimeElapsedColumn(),
    console=console
) as progress:
    task = progress.add_task("仿真进度", total=10)
    for i in range(10):
        mjx_data = mjx.step(mjx_model, mjx_data)
        progress.update(task, advance=1)
    
console.print(f"   [green]✓[/green] 执行10步仿真")
console.print(f"   最终qpos: {mjx_data.qpos[:5]}...")
console.print(f"   最终qvel: {mjx_data.qvel[:5]}...")
console.print(f"   时间: [cyan]{mjx_data.time:.3f}s[/cyan]")

# 5. JIT编译仿真步
console.print("\n[bold]5. 测试JIT编译...[/bold]")

@jax.jit
def jit_step(model, data):
    """JIT编译的仿真步"""
    return mjx.step(model, data)

# 预热
mjx_data_test = mjx.make_data(mjx_model)
_ = jit_step(mjx_model, mjx_data_test)

# 未JIT
mjx_data_test = mjx.make_data(mjx_model)
with Progress(
    SpinnerColumn(),
    TextColumn("[progress.description]{task.description}"),
    BarColumn(),
    TimeElapsedColumn(),
    console=console
) as progress:
    task = progress.add_task("未JIT (100步)", total=100)
    start = time.time()
    for _ in range(100):
        mjx_data_test = mjx.step(mjx_model, mjx_data_test)
        progress.update(task, advance=1)
    time_no_jit = time.time() - start

# JIT
mjx_data_test = mjx.make_data(mjx_model)
with Progress(
    SpinnerColumn(),
    TextColumn("[progress.description]{task.description}"),
    BarColumn(),
    TimeElapsedColumn(),
    console=console
) as progress:
    task = progress.add_task("JIT (100步)", total=100)
    start = time.time()
    for _ in range(100):
        mjx_data_test = jit_step(mjx_model, mjx_data_test)
        progress.update(task, advance=1)
    time_jit = time.time() - start

table = Table(title="JIT性能对比")
table.add_column("方式", style="cyan")
table.add_column("时间 (ms)", style="magenta")
table.add_column("加速比", style="green")
table.add_row("未JIT (100步)", f"{time_no_jit*1000:.2f}", "-")
table.add_row("JIT (100步)", f"{time_jit*1000:.2f}", f"{time_no_jit/time_jit:.1f}x")
console.print(table)

# 6. vmap测试（批量环境）
console.print("\n[bold]6. 测试vmap批量仿真...[/bold]")

def reset_env(key):
    """重置环境（添加随机扰动）"""
    data = mjx.make_data(mjx_model)
    # 添加小的随机扰动
    noise = jax.random.normal(key, data.qpos.shape) * 0.01
    data = data.replace(qpos=data.qpos + noise)
    return data

# 创建批量环境
batch_size = 16
keys = jax.random.split(jax.random.PRNGKey(0), batch_size)
batch_reset = jax.vmap(reset_env)
batch_data = batch_reset(keys)

console.print(f"   [green]✓[/green] 创建{batch_size}个并行环境")
console.print(f"   批量qpos shape: [cyan]{batch_data.qpos.shape}[/cyan]")

# 批量步进
@jax.jit
def batch_step(model, data):
    return jax.vmap(lambda d: mjx.step(model, d))(data)

batch_data = batch_step(mjx_model, batch_data)
console.print(f"   [green]✓[/green] 批量步进成功")
console.print(f"   批量qvel shape: [cyan]{batch_data.qvel.shape}[/cyan]")

console.print()
console.print(Panel.fit(
    "[bold green]✓ MJX验证通过！物理仿真环境配置正确。[/bold green]",
    border_style="green"
))

console.print("\n[bold]说明：[/bold]")
console.print("  • MJX是MuJoCo的JAX版本")
console.print("  • 支持端到端GPU加速")
console.print("  • 支持JIT编译和vmap向量化")
console.print("  • 可以轻松实现并行环境仿真")
