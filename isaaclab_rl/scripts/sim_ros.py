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


def _merge_kit_args(existing: str, additions: list[str]) -> str:
    """合并 kit_args，若用户已显式设置同键则不覆盖。"""
    merged = [item for item in existing.split() if item]
    existing_keys = {item.split("=", 1)[0] for item in merged}
    for item in additions:
        key = item.split("=", 1)[0]
        if key not in existing_keys:
            merged.append(item)
            existing_keys.add(key)
    return " ".join(merged)


# 这些设置必须在 SimulationApp 启动前注入，运行期修改通常已来不及影响 graph prim 包装逻辑。
_OMNIGRAPH_KIT_OVERRIDES = [
    "--/persistent/omnigraph/disablePrimNodes=false",
    "--/app/omnigraph/disablePrimNodes=false",
    "--/persistent/omnigraph/useSchemaPrims=false",
    "--/app/omnigraph/useSchemaPrims=false",
]
app_launcher_args.kit_args = _merge_kit_args(getattr(app_launcher_args, "kit_args", ""), _OMNIGRAPH_KIT_OVERRIDES)
# 对 ROS2 ActionGraph 需求脚本，默认使用 isaaclab.python.kit 更稳定（保留用户显式传入优先）。
if not getattr(app_launcher_args, "experience", ""):
    app_launcher_args.experience = "isaaclab.python.kit"

app_launcher = AppLauncher(app_launcher_args)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import omni.kit.app
import omni.timeline
import torch
import carb

import jiyuan_tasks  # noqa: F401  # 注册环境

TASK_ENV_MAP = {
    "velocity": "Isaac-Jiyuan-Velocity-v0",
    "standing": "Isaac-Jiyuan-Standing-v0",
    "test": "Isaac-Jiyuan-Test-v0",
}

# 延迟导入的 OmniGraph 模块句柄（避免在扩展未启用时导入失败）
og = None


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
        "--ros_domain_id",
        type=int,
        default=int(os.environ.get("ROS_DOMAIN_ID", "0")),
        help="ROS2 Domain ID (默认: 环境变量 ROS_DOMAIN_ID 或 0)",
    )
    parser.add_argument(
        "--tick_source",
        type=str,
        default="impulse",
        choices=["auto", "playback", "physics", "impulse"],
        help="ActionGraph 触发源 (默认: impulse，逐仿真步手动触发)",
    )
    parser.add_argument(
        "--log_every",
        type=int,
        default=200,
        help="每 N 步打印一次统计 (默认: 200)",
    )
    parser.add_argument(
        "--app_update_every_step",
        action="store_true",
        default=True,
        help="每步额外调用 simulation_app.update() 以泵送 OmniGraph/ROS 回调 (默认: 开启)",
    )
    parser.add_argument(
        "--no_app_update_every_step",
        dest="app_update_every_step",
        action="store_false",
        help="关闭每步 simulation_app.update()",
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
                simulation_app.update()
                return ext_name
        except Exception:
            continue
    raise RuntimeError("无法启用 ROS2 Bridge 扩展（isaacsim.ros2.bridge / omni.isaac.ros2_bridge）")


def _enable_extension_candidates(candidates: list[str]) -> str:
    """按候选列表启用扩展，返回成功的扩展名。"""
    ext_manager = omni.kit.app.get_app().get_extension_manager()
    for ext_name in candidates:
        try:
            if ext_manager.is_extension_enabled(ext_name):
                return ext_name
            if ext_manager.set_extension_enabled_immediate(ext_name, True):
                return ext_name
        except Exception:
            continue
    raise RuntimeError(f"无法启用任一扩展: {candidates}")


def _import_omnigraph_core():
    """确保 omni.graph.core 可用并返回模块对象。"""
    enabled_name = _enable_extension_candidates(["omni.graph.core", "omni.graph"])
    # OnPlaybackTick 依赖 omni.graph.action；其余扩展按可用性启用。
    _enable_extension_candidates(["omni.graph.action"])
    for ext_name in ("isaacsim.core.nodes", "omni.isaac.core_nodes"):
        try:
            _enable_extension_candidates([ext_name])
        except Exception:
            pass
    for ext_name in ("omni.graph.nodes", "omni.graph.scriptnode", "omni.graph.ui_nodes"):
        try:
            _enable_extension_candidates([ext_name])
        except Exception:
            pass
    simulation_app.update()
    try:
        return import_module("omni.graph.core"), enabled_name
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            f"已尝试启用 {enabled_name}，但仍无法导入 omni.graph.core。"
            "请确认 Isaac Sim 安装完整且使用 isaaclab.sh 启动。"
        ) from exc


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


def _set_optional_input_attr(node, attr_candidates: list[str], value) -> str | None:
    """尝试设置输入属性（兼容不同节点版本字段名），返回成功的属性名。"""
    for attr_name in attr_candidates:
        try:
            og.Controller.attribute(f"inputs:{attr_name}", node).set(value)
            return attr_name
        except Exception:
            continue
    return None


def _tick_node_candidates(mode: str) -> list[tuple[str, str]]:
    """返回 Tick 节点候选: (node_type, output_attr)。"""
    playback = [("omni.graph.action.OnPlaybackTick", "tick")]
    physics = [
        ("isaacsim.core.nodes.OnPhysicsStep", "step"),
        ("omni.isaac.core_nodes.OnPhysicsStep", "step"),
    ]
    impulse = [("omni.graph.action.OnImpulseEvent", "execOut")]
    if mode == "playback":
        return playback
    if mode == "physics":
        return physics
    if mode == "impulse":
        return impulse
    # auto 模式优先 playback，其次 impulse，最后 physics。
    return playback + impulse + physics


def _ros2_node_prefix(ros2_ext_name: str) -> str:
    """根据启用的 ROS2 Bridge 扩展返回节点类型前缀。"""
    if ros2_ext_name == "omni.isaac.ros2_bridge":
        return "omni.isaac.ros2_bridge"
    return "isaacsim.ros2.bridge"


def _set_stage_edit_target_to_session_layer(stage) -> str:
    """将 Stage 编辑目标切换到 Session Layer，避免只读 Root Layer 写失败。"""
    session_layer = stage.GetSessionLayer()
    if session_layer is None:
        raise RuntimeError("当前 USD Stage 缺少 Session Layer，无法写入 ActionGraph。")
    stage.SetEditTarget(session_layer)
    return session_layer.identifier


def _configure_omnigraph_settings() -> None:
    """在 headless 下放宽 OmniGraph 到 USD 的约束，减少 schema 兼容导致的建图失败。"""
    settings = carb.settings.get_settings()
    # 同时写 /persistent 与 /app 命名空间，适配不同 Kit 版本键路径。
    key_values = [
        ("/persistent/omnigraph/useSchemaPrims", False),
        ("/app/omnigraph/useSchemaPrims", False),
        ("/persistent/omnigraph/disablePrimNodes", False),
        ("/app/omnigraph/disablePrimNodes", False),
    ]
    for key, value in key_values:
        try:
            settings.set(key, value)
        except Exception:
            pass


def _stage_diagnostics(stage) -> str:
    """输出 Stage 编辑诊断信息。"""
    root_layer = stage.GetRootLayer()
    session_layer = stage.GetSessionLayer()
    edit_layer = stage.GetEditTarget().GetLayer()
    root_id = getattr(root_layer, "identifier", "<none>")
    session_id = getattr(session_layer, "identifier", "<none>")
    edit_id = getattr(edit_layer, "identifier", "<none>")
    root_edit = getattr(root_layer, "permissionToEdit", None)
    session_edit = getattr(session_layer, "permissionToEdit", None)
    return (
        f"root={root_id} (editable={root_edit}), "
        f"session={session_id} (editable={session_edit}), "
        f"edit_target={edit_id}"
    )


def _probe_stage_writable(stage) -> tuple[bool, str]:
    """探测当前 Stage 是否可创建普通 Prim。"""
    probe_path = "/__SimRosProbe__"
    try:
        prim = stage.DefinePrim(probe_path, "Xform")
        if not prim.IsValid():
            return False, "DefinePrim 返回无效 Prim"
        stage.RemovePrim(probe_path)
        simulation_app.update()
        return True, "ok"
    except Exception as exc:
        return False, repr(exc)


def _create_ros2_graph_at_path(
    graph_path: str,
    cmd_topic: str,
    fb_topic: str,
    ros2_ext_name: str,
    ros_domain_id: int,
    tick_node_type: str,
    tick_output_attr: str,
    evaluator_name: str,
):
    """在指定路径创建 ROS2 发布/订阅 ActionGraph。"""
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError("当前没有可用 USD Stage，无法创建 ActionGraph。")
    layer_id = _set_stage_edit_target_to_session_layer(stage)

    if stage.GetPrimAtPath(graph_path).IsValid():
        stage.RemovePrim(graph_path)
        simulation_app.update()

    node_prefix = _ros2_node_prefix(ros2_ext_name)
    (_, new_nodes, _, _) = og.Controller.edit(
        {"graph_path": graph_path, "evaluator_name": evaluator_name},
        {
            og.Controller.Keys.CREATE_NODES: [
                ("TickSource", tick_node_type),
                ("RosContext", f"{node_prefix}.ROS2Context"),
                ("CmdPublisher", f"{node_prefix}.ROS2Publisher"),
                ("FbSubscriber", f"{node_prefix}.ROS2Subscriber"),
            ],
            og.Controller.Keys.SET_VALUES: [
                ("CmdPublisher.inputs:topicName", cmd_topic),
                ("FbSubscriber.inputs:topicName", fb_topic),
            ],
            og.Controller.Keys.CONNECT: [
                (f"TickSource.outputs:{tick_output_attr}", "CmdPublisher.inputs:execIn"),
                (f"TickSource.outputs:{tick_output_attr}", "FbSubscriber.inputs:execIn"),
                ("RosContext.outputs:context", "CmdPublisher.inputs:context"),
                ("RosContext.outputs:context", "FbSubscriber.inputs:context"),
            ],
        },
    )
    if len(new_nodes) < 4:
        raise RuntimeError(
            f"ActionGraph 创建返回节点数量异常: len={len(new_nodes)}, "
            f"path={graph_path}, evaluator={evaluator_name}"
        )

    cmd_pub_node = new_nodes[2]
    fb_sub_node = new_nodes[3]
    ros_ctx_node = new_nodes[1]

    domain_attr_name = _set_optional_input_attr(
        ros_ctx_node,
        attr_candidates=["domain_id", "domainId"],
        value=int(ros_domain_id),
    )
    if domain_attr_name is None:
        print("[WARN] ROS2Context 未找到 domain_id/domainId 输入属性，使用节点默认 Domain。", flush=True)
    else:
        print(
            f"[INFO] ROS2Context domain 设置成功: inputs:{domain_attr_name}={int(ros_domain_id)}",
            flush=True,
        )
        use_env_attr_name = _set_optional_input_attr(
            ros_ctx_node,
            attr_candidates=["useDomainIDEnvVar", "use_domain_id_env_var"],
            value=False,
        )
        if use_env_attr_name is not None:
            print(
                f"[INFO] ROS2Context domain 来源固定为节点输入: inputs:{use_env_attr_name}=False",
                flush=True,
            )

    _set_dynamic_message_type(cmd_pub_node, message_package="std_msgs", message_name="Float32MultiArray")
    _set_dynamic_message_type(fb_sub_node, message_package="std_msgs", message_name="Float32MultiArray")

    # MultiArray 结构字段（最小有效配置）
    og.Controller.attribute("inputs:layout:data_offset", cmd_pub_node).set(0)
    og.Controller.attribute("inputs:layout:dim", cmd_pub_node).set([])

    impulse_attr_path = None
    if tick_node_type.endswith("OnImpulseEvent"):
        impulse_attr_path = f"{graph_path}/TickSource.state:enableImpulse"

    return cmd_pub_node, fb_sub_node, layer_id, impulse_attr_path


def _build_ros2_graph(
    cmd_topic: str,
    fb_topic: str,
    ros2_ext_name: str,
    ros_domain_id: int,
    tick_source: str,
):
    """创建 ROS2 发布/订阅 ActionGraph（带路径回退）。"""
    import omni.usd

    _configure_omnigraph_settings()
    candidates = [
        "/World/SimRosBridgeGraph",  # 优先放到 World 下，减少根路径冲突
        "/SimRosBridgeGraph",        # 根路径独立图
        "/ActionGraph",              # Isaac 常用路径
        "/Ros2BridgeGraph",          # 最后兜底
    ]
    tick_candidates = _tick_node_candidates(tick_source)

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError("当前没有可用 USD Stage，无法创建 ActionGraph。")
    print(f"[INFO] Stage 诊断: {_stage_diagnostics(stage)}", flush=True)
    writable, reason = _probe_stage_writable(stage)
    print(f"[INFO] Stage 可写探针: writable={writable}, reason={reason}", flush=True)

    last_error = None
    for tick_node_type, tick_output_attr in tick_candidates:
        # OnPhysicsStep 需要 on-demand 图，优先尝试 push evaluator。
        if "OnPhysicsStep" in tick_node_type:
            evaluator_candidates = ["push", "execution"]
        else:
            evaluator_candidates = ["execution", "push"]
        for evaluator_name in evaluator_candidates:
            for graph_path in candidates:
                try:
                    cmd_pub_node, fb_sub_node, layer_id, impulse_attr_path = _create_ros2_graph_at_path(
                        graph_path=graph_path,
                        cmd_topic=cmd_topic,
                        fb_topic=fb_topic,
                        ros2_ext_name=ros2_ext_name,
                        ros_domain_id=ros_domain_id,
                        tick_node_type=tick_node_type,
                        tick_output_attr=tick_output_attr,
                        evaluator_name=evaluator_name,
                    )
                    print(
                        "[INFO] ROS2 ActionGraph 创建成功: "
                        f"{graph_path} (tick={tick_node_type}, evaluator={evaluator_name}, edit_target={layer_id})",
                        flush=True,
                    )
                    return cmd_pub_node, fb_sub_node, tick_node_type, impulse_attr_path
                except Exception as exc:
                    last_error = exc
                    print(
                        "[WARN] ROS2 ActionGraph 创建失败("
                        f"{graph_path}, tick={tick_node_type}, evaluator={evaluator_name}"
                        f"): {exc}",
                        flush=True,
                    )

    raise RuntimeError(f"无法创建 ROS2 ActionGraph，候选路径均失败: {candidates}") from last_error


def _trigger_impulse_tick_if_needed(impulse_attr_path: str | None) -> None:
    """若使用 OnImpulseEvent，则触发一次图执行。"""
    if not impulse_attr_path:
        return
    og.Controller.set(og.Controller.attribute(impulse_attr_path), True)


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
    global og
    args = parse_args()

    if not os.path.isabs(args.usd_path):
        raise ValueError(f"--usd_path 必须是绝对路径: {args.usd_path}")
    usd_path = os.path.abspath(args.usd_path)
    if not os.path.exists(usd_path):
        raise ValueError(f"--usd_path 指向文件不存在: {usd_path}")

    os.environ["JIYUAN_USD_PATH"] = usd_path

    enabled_ext = _enable_ros2_bridge_extension()
    og, og_ext_name = _import_omnigraph_core()
    if args.tick_source == "physics":
        print(
            "[WARN] tick_source=physics 在当前图配置下可能不触发（需要 on-demand graph）。"
            "已自动回退到 impulse。",
            flush=True,
        )
        args.tick_source = "impulse"
    print("=" * 80)
    print("Isaac Headless ROS2 联调入口")
    print("=" * 80)
    print(f"任务: {args.task}")
    print(f"环境ID: {TASK_ENV_MAP[args.task]}")
    print(f"USD 路径: {usd_path}")
    print(f"ROS2 扩展: {enabled_ext}")
    print(f"OmniGraph 扩展: {og_ext_name}")
    print(f"命令话题: {args.cmd_topic}")
    print(f"反馈话题: {args.fb_topic}")
    print(f"ROS_DOMAIN_ID: {args.ros_domain_id}")
    print(f"Tick Source: {args.tick_source}")
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

        cmd_pub_node, fb_sub_node, tick_node_type, impulse_attr_path = _build_ros2_graph(
            cmd_topic=args.cmd_topic,
            fb_topic=args.fb_topic,
            ros2_ext_name=enabled_ext,
            ros_domain_id=args.ros_domain_id,
            tick_source=args.tick_source,
        )
        if impulse_attr_path is not None:
            print(f"[INFO] 使用 OnImpulseEvent 手动触发: {impulse_attr_path}")
        print(f"[INFO] 当前 Tick 节点: {tick_node_type}")
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
            _trigger_impulse_tick_if_needed(impulse_attr_path)
            pub_count += 1

            # 驱动环境动作（按 action_dim 截断/填零）
            actions[:] = 0.0
            usable_dim = min(action_dim, 16)
            cmd_t = torch.as_tensor(cmd[:usable_dim], device=args.device).view(1, usable_dim)
            actions[:, :usable_dim] = cmd_t

            obs, _, _, _, _ = env.step(actions)
            _ = obs
            if args.app_update_every_step:
                simulation_app.update()

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
                if fb_count == 0 and (step_count + 1) >= max(200, int(args.log_every)):
                    print(
                        "[WARN] 目前未收到任何反馈消息。请检查："
                        "1) 是否有节点在发布反馈话题；"
                        f"2) ROS_DOMAIN_ID 是否一致(当前={args.ros_domain_id})；"
                        "3) 反馈消息类型是否为 std_msgs/Float32MultiArray(16维)。"
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
