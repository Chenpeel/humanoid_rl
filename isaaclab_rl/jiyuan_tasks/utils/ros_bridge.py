"""
Isaac 侧 ROS 舵机桥接工具。

功能:
1. 发布 `ServoCommand` 到 ROS 舵机控制话题
2. 订阅 `ServoState` 并缓存最新状态
3. 复用 `ParallelAnkleMapper`，将策略动作转换为舵机命令
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any, Dict, Iterable, Mapping, Optional

import numpy as np

from .config_loader import load_default_robot_config, load_robot_config
from .sim2real import ParallelAnkleMapper

try:  # pragma: no cover - torch 为可选依赖（用于 Tensor 检查）
    import torch
except Exception:  # pragma: no cover
    torch = None

ROS_IMPORT_ERROR: Exception | None = None

try:  # pragma: no cover - 在无 ROS 环境下允许导入模块
    import rclpy
    from rclpy.node import Node
    from servo_msgs.msg import ServoCommand, ServoState
except Exception as exc:  # pragma: no cover
    rclpy = None
    Node = Any
    ServoCommand = Any
    ServoState = Any
    ROS_IMPORT_ERROR = exc


@dataclass(frozen=True, slots=True)
class ServoCommandEntry:
    """舵机命令的统一中间表示。"""

    servo_type: str
    servo_id: int
    position: int
    speed: int


@dataclass(frozen=True, slots=True)
class ServoStateEntry:
    """舵机状态缓存结构。"""

    servo_type: str
    servo_id: int
    position: int
    load: int
    temperature: int
    error_code: int
    stamp_sec: float
    received_time_sec: float


def ensure_ros_available() -> None:
    """确保 ROS 依赖可用。"""
    if rclpy is None:
        raise RuntimeError(
            "未检测到 ROS 2 运行时依赖（rclpy / servo_msgs）。"
            "请在已 source ROS 环境后再启用 --ros_bridge。"
        ) from ROS_IMPORT_ERROR


def _clamp_uint16(value: int | float) -> int:
    return max(0, min(65535, int(round(float(value)))))


def _to_action_array(action: Any) -> np.ndarray:
    """将 Tensor/数组/列表统一转成 1D numpy 数组。"""
    if torch is not None and isinstance(action, torch.Tensor):
        array = action.detach().cpu().numpy()
    else:
        array = np.asarray(action)

    if array.ndim != 1:
        raise ValueError(f"动作必须为 1D 向量，当前形状: {array.shape}")

    return array.astype(np.float64, copy=False)


def _require_mapping(container: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(container, Mapping):
        raise ValueError(f"配置项 {path} 必须为字典结构，当前类型: {type(container)}")
    return container


def build_ankle_indices(robot_config: Mapping[str, Any]) -> Dict[str, list[int]]:
    """从 robot_config 构建 `ParallelAnkleMapper` 所需的索引结构。"""
    action_mapping = _require_mapping(robot_config.get("action_mapping"), "action_mapping")
    ankle_indices = _require_mapping(action_mapping.get("ankle_indices"), "action_mapping.ankle_indices")
    left = _require_mapping(ankle_indices.get("left"), "action_mapping.ankle_indices.left")
    right = _require_mapping(ankle_indices.get("right"), "action_mapping.ankle_indices.right")

    return {
        "left": [int(left["roll"]), int(left["pitch"]), int(left["yaw"])],
        "right": [int(right["roll"]), int(right["pitch"]), int(right["yaw"])],
    }


def create_parallel_ankle_mapper(robot_config_path: str | Path | None = None) -> ParallelAnkleMapper:
    """根据机器人配置创建并联脚踝映射器。"""
    if robot_config_path:
        robot_cfg = load_robot_config(robot_config_path)
    else:
        robot_cfg = load_default_robot_config()

    parallel_ankle_cfg = _require_mapping(robot_cfg.get("parallel_ankle"), "parallel_ankle")
    sim2real_cfg = robot_cfg.get("sim2real") or {}
    sim2real_cfg = _require_mapping(sim2real_cfg, "sim2real")
    filtering_cfg = _require_mapping(sim2real_cfg.get("filtering") or {}, "sim2real.filtering")

    return ParallelAnkleMapper(
        ankle_indices=build_ankle_indices(robot_cfg),
        l0=float(parallel_ankle_cfg["l0"]),
        l1=float(parallel_ankle_cfg["l1"]),
        l2=float(parallel_ankle_cfg["l2"]),
        enable_filtering=bool(filtering_cfg.get("enable", True)),
        filter_alpha=float(filtering_cfg.get("alpha", 0.7)),
    )


def _coerce_int(value: Any, field_name: str) -> int:
    try:
        return int(round(float(value)))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"字段 {field_name} 无法转为整数: {value}") from exc


def prepare_servo_command_entries(
    commands: Iterable[Mapping[str, Any]],
    default_speed: int = 100,
    servo_type: str = "bus",
    speed_override: int | None = None,
) -> list[ServoCommandEntry]:
    """将 solver 输出命令转换为标准命令结构。"""
    result: list[ServoCommandEntry] = []
    for command in commands:
        raw_servo_id = command.get("servo_id", command.get("id"))
        if raw_servo_id is None:
            raise ValueError(f"命令缺少 servo_id/id 字段: {command}")

        raw_position = command.get("position")
        if raw_position is None:
            raise ValueError(f"命令缺少 position 字段: {command}")

        raw_speed = speed_override if speed_override is not None else command.get("speed", default_speed)

        result.append(
            ServoCommandEntry(
                servo_type=str(command.get("servo_type", servo_type)),
                servo_id=_clamp_uint16(_coerce_int(raw_servo_id, "servo_id")),
                position=_clamp_uint16(_coerce_int(raw_position, "position")),
                speed=_clamp_uint16(_coerce_int(raw_speed, "speed")),
            )
        )

    return result


class IsaacServoRosBridge:
    """Isaac 侧 ROS 舵机桥接器。"""

    def __init__(
        self,
        node_name: str = "isaac_ros_bridge",
        command_topic: str = "/sim/servo_command",
        state_topic: str = "/sim/servo_state",
        default_speed: int = 100,
        servo_type: str = "bus",
        qos_depth: int = 10,
        auto_start: bool = False,
    ):
        self.node_name = node_name
        self.command_topic = command_topic
        self.state_topic = state_topic
        self.default_speed = int(default_speed)
        self.servo_type = servo_type
        self.qos_depth = int(qos_depth)

        self._node: Node | None = None
        self._command_pub = None
        self._state_sub = None
        self._owns_rclpy_context = False
        self._state_cache: Dict[int, ServoStateEntry] = {}

        if auto_start:
            self.start()

    def start(self) -> None:
        """初始化 ROS 节点与话题通信。"""
        if self._node is not None:
            return

        ensure_ros_available()
        assert rclpy is not None

        if not rclpy.ok():
            rclpy.init(args=None)
            self._owns_rclpy_context = True

        self._node = rclpy.create_node(self.node_name)
        self._command_pub = self._node.create_publisher(ServoCommand, self.command_topic, self.qos_depth)
        self._state_sub = self._node.create_subscription(ServoState, self.state_topic, self._on_state_msg, self.qos_depth)
        self._node.get_logger().info(
            f"IsaacServoRosBridge 已启动: command={self.command_topic}, state={self.state_topic}"
        )

    def close(self) -> None:
        """释放 ROS 资源。"""
        if self._node is not None:
            self._node.destroy_node()
            self._node = None
            self._command_pub = None
            self._state_sub = None

        if self._owns_rclpy_context and rclpy is not None and rclpy.ok():
            rclpy.shutdown()
            self._owns_rclpy_context = False

    def __enter__(self) -> "IsaacServoRosBridge":
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def _require_started(self) -> None:
        if self._node is None or self._command_pub is None:
            raise RuntimeError("IsaacServoRosBridge 尚未启动，请先调用 start()")

    def spin_once(self, timeout_sec: float = 0.0) -> None:
        """推进一次 ROS 回调。"""
        if self._node is None or rclpy is None:
            return
        rclpy.spin_once(self._node, timeout_sec=max(0.0, float(timeout_sec)))

    def _on_state_msg(self, msg: ServoState) -> None:
        stamp_sec = 0.0
        stamp = getattr(msg, "stamp", None)
        if stamp is not None and hasattr(stamp, "sec") and hasattr(stamp, "nanosec"):
            stamp_sec = float(stamp.sec) + float(stamp.nanosec) * 1e-9

        state = ServoStateEntry(
            servo_type=str(msg.servo_type),
            servo_id=int(msg.servo_id),
            position=int(msg.position),
            load=int(msg.load),
            temperature=int(msg.temperature),
            error_code=int(msg.error_code),
            stamp_sec=stamp_sec,
            received_time_sec=time.monotonic(),
        )
        self._state_cache[state.servo_id] = state

    def publish_servo_command(
        self,
        servo_id: int,
        position: int,
        speed: int | None = None,
        servo_type: str | None = None,
    ) -> None:
        """发布单条舵机命令。"""
        self._require_started()
        assert self._node is not None

        msg = ServoCommand()
        msg.servo_type = str(servo_type or self.servo_type)
        msg.servo_id = _clamp_uint16(servo_id)
        msg.position = _clamp_uint16(position)
        msg.speed = _clamp_uint16(self.default_speed if speed is None else speed)
        msg.stamp = self._node.get_clock().now().to_msg()
        self._command_pub.publish(msg)

    def publish_sim2real_commands(
        self,
        commands: Iterable[Mapping[str, Any]],
        speed_override: int | None = None,
    ) -> list[ServoCommandEntry]:
        """发布由 solver/sim2real 生成的命令列表。"""
        entries = prepare_servo_command_entries(
            commands=commands,
            default_speed=self.default_speed,
            servo_type=self.servo_type,
            speed_override=speed_override,
        )

        for entry in entries:
            self.publish_servo_command(
                servo_id=entry.servo_id,
                position=entry.position,
                speed=entry.speed,
                servo_type=entry.servo_type,
            )

        return entries

    def publish_action(
        self,
        action: Any,
        mapper: ParallelAnkleMapper,
        speed_override: int | None = None,
    ) -> list[ServoCommandEntry]:
        """将策略动作映射为舵机命令并发布。"""
        action_array = _to_action_array(action)
        left_commands = mapper.map_action(action_array, "left")
        right_commands = mapper.map_action(action_array, "right")
        return self.publish_sim2real_commands([*left_commands, *right_commands], speed_override=speed_override)

    def get_state(self, servo_id: int) -> ServoStateEntry | None:
        """获取指定舵机的最新状态。"""
        return self._state_cache.get(int(servo_id))

    def get_latest_states(self) -> Dict[int, ServoStateEntry]:
        """获取全部舵机最新状态快照。"""
        return dict(self._state_cache)


__all__ = [
    "ServoCommandEntry",
    "ServoStateEntry",
    "IsaacServoRosBridge",
    "ensure_ros_available",
    "build_ankle_indices",
    "create_parallel_ankle_mapper",
    "prepare_servo_command_entries",
]
