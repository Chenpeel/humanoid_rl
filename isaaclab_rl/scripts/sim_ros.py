#!/usr/bin/env python3
"""
Jiyuan Isaac Headless ROS2 联调入口（无 rclpy）。

目标：
1. 在 SSH/headless 下运行 USD 仿真
2. 使用 Isaac ROS2 Bridge Core 发布 `/sim/joint_cmd`（Float32MultiArray, 16 维）
3. 使用 Isaac ROS2 Bridge Core 订阅 `/sim/joint_state_fb`（Float32MultiArray, 16 维）
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from importlib import import_module
from pathlib import Path

# 添加项目根目录到 Python 路径
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

# 必须先启动 Isaac Sim 应用（在导入 Isaac Lab 之前）
from isaaclab.app import AppLauncher

# 先解析 AppLauncher 参数并启动应用
app_launcher_parser = argparse.ArgumentParser(add_help=False)
AppLauncher.add_app_launcher_args(app_launcher_parser)
app_launcher_args, _ = app_launcher_parser.parse_known_args()
app_launcher = AppLauncher(app_launcher_args)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import omni.graph.core as og
import omni.kit.app
import omni.timeline
import torch

import jiyuan_tasks  # noqa: F401  # 注册环境

TASK_ENV_MAP = {
    "velocity": "Isaac-Jiyuan-Velocity-v0",
    "standing": "Isaac-Jiyuan-Standing-v0",
    "test": "Isaac-Jiyuan-Test-v0",
}


def parse_args():
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="Isaac headless ROS2 联调入口（Bridge Core）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        parents=[app_launcher_parser],
    )

    parser.add_argument(
        "--task",
        type=str,
        default="standing",
        choices=list(TASK_ENV_MAP.keys()),
        help="环境任务 (默认: standing)",
    )
    parser.add_argument(
        "--usd_path",
        type=str,
        required=True,
        help="USD 绝对路径（必选）",
    )
    parser.add_argument(
        "--sim_steps",
        type=int,
        default=20000,
        help="仿真总步数 (默认: 20000)",
    )
    parser.add_argument(
        "--action_source",
        type=str,
        default="zero",
        choices=["zero", "sine"],
        help="动作来源 (默认: zero)",
    )
    parser.add_argument(
        "--sine_amp",
        type=float,
        default=0.2,
        help="正弦动作幅值（rad）(默认: 0.2)",
    )
    parser.add_argument(
        "--sine_freq",
        type=float,
        default=0.5,
        help="正弦动作频率（Hz）(默认: 0.5)",
    )
    parser.add_argument(
        "--cmd_topic",
        type=str,
        default="/sim/joint_cmd",
        help="命令话题 (默认: /sim/joint_cmd)",
    )
    parser.add_argument(
        "--fb_topic",
        type=str,
        default="/sim/joint_state_fb",
        help="反馈话题 (默认: /sim/joint_state_fb)",
    )
    parser.add_argument(
        "--num_envs",
        type=int,
        default=1,
        help="并行环境数 (默认: 1)",
    )
    parser.add_argument(
        "--log_every",
        type=int,
        default=200,
        help="每 N 步打印一次统计 (默认: 200)",
    )

    return parser.parse_args()


def _enable_ros2_bridge_extension() -> str:
    """启用 ROS2 Bridge 扩展。"""
    ext_manager = omni.kit.app.get_app().get_extension_manager()
    candidates = ["isaacsim.ros2.bridge", "omni.isaac.ros2_bridge"]
    for ext_name in candidates:
        try:
            if ext_manager.is_extension_enabled(ext_name):
                return ext_name
            if ext_manager.set_extension_enabled_immediate(ext_name, True):
                return ext_name
        except Exception:
            continue
    raise RuntimeError("无法启用 ROS2 Bridge 扩展（isaacsim.ros2.bridge / omni.isaac.ros2_bridge）")


def _build_env_cfg(task_id: str, num_envs: int):
    """从 Gym 注册信息解析并实例化 env cfg。"""
    env_spec = gym.spec(task_id)
    env_cfg_entry_point = env_spec.kwargs["env_cfg_entry_point"]
    module_path, obj_name = env_cfg_entry_point.rsplit(":", 1)
    module = import_module(module_path)
    env_cfg = getattr(module, obj_name)
    if isinstance(env_cfg, type):
        env_cfg = env_cfg()
    env_cfg.scene.num_envs = int(num_envs)
    return env_cfg


def _set_dynamic_message_type(ogn_node, message_package: str, message_name: str, message_subfolder: str = "msg") -> None:
    """按 ROS2 Bridge 官方测试流程设置动态消息类型。"""
    og.Controller.attribute("inputs:messageName", ogn_node).set("")
    simulation_app.update()
    og.Controller.attribute("inputs:messagePackage", ogn_node).set(message_package)
    og.Controller.attribute("inputs:messageSubfolder", ogn_node).set(message_subfolder)
    og.Controller.attribute("inputs:messageName", ogn_node).set(message_name)
    simulation_app.update()


def _build_ros2_graph(cmd_topic: str, fb_topic: str):
    """创建 ROS2 发布/订阅 ActionGraph。"""
    import omni.usd

    graph_path = "/ActionGraph/SimRosBridge"
    stage = omni.usd.get_context().get_stage()
    if stage.GetPrimAtPath(graph_path).IsValid():
        stage.RemovePrim(graph_path)
        simulation_app.update()

    (_, new_nodes, _, _) = og.Controller.edit(
        {"graph_path": graph_path, "evaluator_name": "execution"},
        {
            og.Controller.Keys.CREATE_NODES: [
                ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                ("RosContext", "isaacsim.ros2.bridge.ROS2Context"),
                ("CmdPublisher", "isaacsim.ros2.bridge.ROS2Publisher"),
                ("FbSubscriber", "isaacsim.ros2.bridge.ROS2Subscriber"),
            ],
            og.Controller.Keys.SET_VALUES: [
                ("CmdPublisher.inputs:topicName", cmd_topic),
                ("FbSubscriber.inputs:topicName", fb_topic),
            ],
            og.Controller.Keys.CONNECT: [
                ("OnPlaybackTick.outputs:tick", "CmdPublisher.inputs:execIn"),
                ("OnPlaybackTick.outputs:tick", "FbSubscriber.inputs:execIn"),
                ("RosContext.outputs:context", "CmdPublisher.inputs:context"),
                ("RosContext.outputs:context", "FbSubscriber.inputs:context"),
            ],
        },
    )

    cmd_pub_node = new_nodes[2]
    fb_sub_node = new_nodes[3]

    _set_dynamic_message_type(cmd_pub_node, message_package="std_msgs", message_name="Float32MultiArray")
    _set_dynamic_message_type(fb_sub_node, message_package="std_msgs", message_name="Float32MultiArray")

    # MultiArray 结构字段（最小有效配置）
    og.Controller.attribute("inputs:layout:data_offset", cmd_pub_node).set(0)
    og.Controller.attribute("inputs:layout:dim", cmd_pub_node).set([])

    return cmd_pub_node, fb_sub_node


def _build_command_vector(step_count: int, action_source: str, sine_amp: float, sine_freq: float) -> np.ndarray:
    """生成 16 维关节命令向量（rad）。"""
    cmd = np.zeros(16, dtype=np.float32)
    if action_source == "sine":
        t = step_count / 50.0
        value = float(sine_amp) * math.sin(2.0 * math.pi * float(sine_freq) * t)
        cmd[9] = value
        cmd[10] = -value
        cmd[12] = value
        cmd[13] = -value
    return cmd


def main():
    args = parse_args()

    if not os.path.isabs(args.usd_path):
        raise ValueError(f"--usd_path 必须是绝对路径: {args.usd_path}")
    usd_path = os.path.abspath(args.usd_path)
    if not os.path.exists(usd_path):
        raise ValueError(f"--usd_path 指向文件不存在: {usd_path}")

    os.environ["JIYUAN_USD_PATH"] = usd_path

    enabled_ext = _enable_ros2_bridge_extension()
    print("=" * 80)
    print("Isaac Headless ROS2 联调入口")
    print("=" * 80)
    print(f"任务: {args.task}")
    print(f"环境ID: {TASK_ENV_MAP[args.task]}")
    print(f"USD 路径: {usd_path}")
    print(f"ROS2 扩展: {enabled_ext}")
    print(f"命令话题: {args.cmd_topic}")
    print(f"反馈话题: {args.fb_topic}")
    print(f"总步数: {args.sim_steps}")
    print(f"动作源: {args.action_source}")
    print("=" * 80)

    env = None
    try:
        env_cfg = _build_env_cfg(TASK_ENV_MAP[args.task], args.num_envs)
        env = gym.make(
            TASK_ENV_MAP[args.task],
            cfg=env_cfg,
            render_mode=None,
        )

        num_envs = int(getattr(env.unwrapped, "num_envs", args.num_envs))
        action_shape = getattr(env.action_space, "shape", None)
        if not action_shape:
            raise ValueError(f"动作空间缺少 shape: {env.action_space}")
        action_dim = int(action_shape[-1])

        print(f"[INFO] 环境创建成功: num_envs={num_envs}, action_dim={action_dim}")
        if action_dim < 16:
            print(f"[WARN] action_dim={action_dim} < 16，发布仍为16维，环境动作将按可用维度截断。")

        cmd_pub_node, fb_sub_node = _build_ros2_graph(args.cmd_topic, args.fb_topic)
        timeline = omni.timeline.get_timeline_interface()
        timeline.play()

        obs, _ = env.reset()
        _ = obs
        actions = torch.zeros((num_envs, action_dim), device=args.device)
        last_feedback = np.zeros(16, dtype=np.float32)

        pub_count = 0
        fb_count = 0
        fb_invalid = 0

        for step_count in range(int(args.sim_steps)):
            cmd = _build_command_vector(
                step_count=step_count,
                action_source=args.action_source,
                sine_amp=args.sine_amp,
                sine_freq=args.sine_freq,
            )

            # 发布 16 维命令到 ROS2
            og.Controller.attribute("inputs:data", cmd_pub_node).set(cmd.tolist())
            pub_count += 1

            # 驱动环境动作（按 action_dim 截断/填零）
            actions[:] = 0.0
            usable_dim = min(action_dim, 16)
            cmd_t = torch.as_tensor(cmd[:usable_dim], device=args.device).view(1, usable_dim)
            actions[:, :usable_dim] = cmd_t

            obs, _, _, _, _ = env.step(actions)
            _ = obs

            # 拉取最新反馈
            fb_data = og.Controller.attribute("outputs:data", fb_sub_node).get()
            if fb_data is not None:
                fb_np = np.asarray(fb_data, dtype=np.float32).reshape(-1)
                if fb_np.size == 16 and np.all(np.isfinite(fb_np)):
                    last_feedback = fb_np
                    fb_count += 1
                elif fb_np.size > 0:
                    fb_invalid += 1

            if (step_count + 1) % max(1, int(args.log_every)) == 0:
                print(
                    f"[INFO] step={step_count + 1}/{args.sim_steps} "
                    f"pub={pub_count} fb_ok={fb_count} fb_invalid={fb_invalid} "
                    f"fb_head={np.array2string(last_feedback[:4], precision=4)}"
                )

        print("=" * 80)
        print("[DONE] 仿真完成")
        print(f"总步数: {args.sim_steps}")
        print(f"发布次数: {pub_count}")
        print(f"反馈有效次数: {fb_count}")
        print(f"反馈无效次数: {fb_invalid}")
        print("=" * 80)
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
