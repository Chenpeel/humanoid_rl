"""
约束性能测试脚本
测试不同约束配置对仿真速度的影响
"""

import time
import jax
import jax.numpy as jp
import mujoco
from mujoco import mjx
from pathlib import Path
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

console = Console()

def create_model_with_constraints(constraint_config):
    """创建带有不同约束配置的模型"""
    
    # 基础XML模板
    base_xml = """<?xml version='1.0' encoding='utf-8' ?>
<mujoco model="test_constraints">
    <visual>
        <quality shadowsize="2048" />
    </visual>
    
    <default>
        <default class="motor">
            <joint />
            <motor />
        </default>
        <default class="collision">
            <geom
                condim="3"
                contype="0"
                conaffinity="1"
                priority="1"
                group="1"
                solref="0.005 1"
                solimp="0.99 0.999 1e-05"
                friction="1 0.01 0.01"
                mass="0"
            />
        </default>
        <default class="constraint_{type}">
            <equality solref="{solref}" solimp="{solimp}" />
        </default>
    </default>
    
    <worldbody>
        <light pos="0 0 3" dir="0 0 -1" diffuse="0.8 0.8 0.8" specular="0.2 0.2 0.2" directional="true" />
        <geom type="plane" size="10 10 0.1" pos="0 0 0" condim="3" friction="1 0.005 0.0001" />
        
        <body name="base" pos="0 0 0.5">
            <freejoint name="floating_base" />
            <geom type="box" size="0.1 0.1 0.1" mass="1.0" />
            
            <!-- 第一个关节链 -->
            <body name="link1" pos="0.2 0 0">
                <joint name="joint1" type="hinge" axis="0 0 1" range="-3.14 3.14" />
                <geom type="box" size="0.05 0.05 0.2" mass="0.5" />
                
                <body name="link1_end" pos="0 0 0.2">
                    <joint name="joint1_end" type="hinge" axis="0 0 1" range="-3.14 3.14" />
                    <geom type="sphere" size="0.03" mass="0.1" />
                </body>
            </body>
            
            <!-- 第二个关节链 -->
            <body name="link2" pos="-0.2 0 0">
                <joint name="joint2" type="hinge" axis="0 0 1" range="-3.14 3.14" />
                <geom type="box" size="0.05 0.05 0.2" mass="0.5" />
                
                <body name="link2_end" pos="0 0 0.2">
                    <joint name="joint2_end" type="hinge" axis="0 0 1" range="-3.14 3.14" />
                    <geom type="sphere" size="0.03" mass="0.1" />
                </body>
            </body>
        </body>
    </worldbody>
    
    <equality>
        <!-- 添加约束 -->
        {constraints}
    </equality>
    
    <actuator>
        <motor name="motor1" joint="joint1" class="motor" />
        <motor name="motor1_end" joint="joint1_end" class="motor" />
        <motor name="motor2" joint="joint2" class="motor" />
        <motor name="motor2_end" joint="joint2_end" class="motor" />
    </actuator>
</mujoco>"""
    
    # 约束配置
    constraints_xml = ""
    if constraint_config["type"] == "none":
        constraints_xml = "<!-- 无约束 -->"
    elif constraint_config["type"] == "tight":
        constraints_xml = """
        <joint
            joint1="joint1"
            joint2="joint2"
            polycoef="0 1 0 0 0"
            class="constraint_tight"
        />
        <joint
            joint1="joint1_end"
            joint2="joint2_end"
            polycoef="0 1 0 0 0"
            class="constraint_tight"
        />"""
    elif constraint_config["type"] == "loose":
        constraints_xml = """
        <joint
            joint1="joint1"
            joint2="joint2"
            polycoef="0 1 0 0 0"
            class="constraint_loose"
        />
        <joint
            joint1="joint1_end"
            joint2="joint2_end"
            polycoef="0 1 0 0 0"
            class="constraint_loose"
        />"""
    elif constraint_config["type"] == "complex":
        constraints_xml = """
        <joint
            joint1="joint1"
            joint2="joint2"
            polycoef="0 1 0.1 0.01 0.001"
            class="constraint_tight"
        />
        <joint
            joint1="joint1_end"
            joint2="joint2_end"
            polycoef="0 0.8 0.2 0.05 0.01"
            class="constraint_loose"
        />"""
    
    # 填充XML
    xml_content = base_xml.format(
        type=constraint_config["type"],
        solref=constraint_config["solref"],
        solimp=constraint_config["solimp"],
        constraints=constraints_xml
    )
    
    # 创建模型
    model = mujoco.MjModel.from_xml_string(xml_content)
    model.opt.timestep = 0.002
    
    return model

def benchmark_simulation(model, steps=1000, warmup=100):
    """基准测试仿真速度"""
    
    # 转换为MJX模型
    mjx_model = mjx.put_model(model)
    
    # 创建初始数据
    data = mjx.make_data(mjx_model)
    
    # 预热
    for i in range(warmup):
        data = mjx.step(mjx_model, data)
    
    # 正式测试
    start_time = time.time()
    for i in range(steps):
        data = mjx.step(mjx_model, data)
    
    elapsed = time.time() - start_time
    
    # 计算性能指标
    sim_time = steps * model.opt.timestep
    real_time_factor = sim_time / elapsed if elapsed > 0 else 0
    steps_per_second = steps / elapsed if elapsed > 0 else 0
    
    return {
        "elapsed_time": elapsed,
        "sim_time": sim_time,
        "real_time_factor": real_time_factor,
        "steps_per_second": steps_per_second,
        "constraint_count": model.neq if hasattr(model, 'neq') else 0
    }

def test_dynamic_constraint_changes():
    """测试动态修改约束参数的性能影响"""
    
    console.print("\n[bold cyan]测试动态约束修改性能[/bold cyan]")
    
    # 创建基础模型（紧约束）
    model = create_model_with_constraints({
        "type": "tight",
        "solref": "0.001 1",
        "solimp": "0.99 0.999