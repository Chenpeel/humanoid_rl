"""
配置加载工具

提供统一的配置加载接口，支持配置优先级：命令行参数 > 配置文件 > 代码预定义值

使用示例:
    from jiyuan_tasks.utils.config_loader import load_train_config, load_robot_config

    # 加载训练配置（支持命令行覆盖）
    cfg = load_train_config("configs/train_config.yaml", cli_overrides={"ppo.algorithm.learning_rate": 0.0005})

    # 加载机器人配置
    robot_cfg = load_robot_config("configs/robot_config.yaml")
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional, Union
import yaml


class ConfigDict(dict):
    """支持点号访问的字典类

    示例:
        cfg = ConfigDict({"a": {"b": 1}})
        print(cfg.a.b)  # 输出: 1
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for key, value in self.items():
            if isinstance(value, dict):
                self[key] = ConfigDict(value)

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"'ConfigDict' object has no attribute '{key}'")

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        try:
            del self[key]
        except KeyError:
            raise AttributeError(f"'ConfigDict' object has no attribute '{key}'")


def load_yaml_config(config_path: Union[str, Path]) -> ConfigDict:
    """加载 YAML 配置文件

    Args:
        config_path: 配置文件路径

    Returns:
        ConfigDict: 配置字典

    Raises:
        FileNotFoundError: 配置文件不存在
        yaml.YAMLError: YAML 格式错误
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        try:
            config = yaml.safe_load(f)
            return ConfigDict(config)
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"YAML 格式错误: {config_path}\n{e}")


def apply_cli_overrides(config: ConfigDict, overrides: Optional[Dict[str, Any]] = None) -> ConfigDict:
    """应用命令行覆盖参数

    支持嵌套键访问，例如: "ppo.algorithm.learning_rate" -> config["ppo"]["algorithm"]["learning_rate"]

    Args:
        config: 基础配置
        overrides: 命令行覆盖字典，键使用点号分隔嵌套路径

    Returns:
        ConfigDict: 应用覆盖后的配置

    示例:
        config = ConfigDict({"ppo": {"algorithm": {"learning_rate": 0.001}}})
        overrides = {"ppo.algorithm.learning_rate": 0.0005}
        config = apply_cli_overrides(config, overrides)
        print(config.ppo.algorithm.learning_rate)  # 输出: 0.0005
    """
    if overrides is None:
        return config

    for key, value in overrides.items():
        keys = key.split(".")
        current = config

        # 导航到嵌套字典的最后一层
        for k in keys[:-1]:
            if k not in current:
                current[k] = ConfigDict()
            current = current[k]

        # 设置最终值
        current[keys[-1]] = value
        print(f"✓ 命令行覆盖: {key} = {value}")

    return config


def load_train_config(
    config_path: Union[str, Path],
    cli_overrides: Optional[Dict[str, Any]] = None,
    defaults: Optional[Dict[str, Any]] = None,
) -> ConfigDict:
    """加载训练配置（支持三级优先级）

    优先级: 命令行参数 > 配置文件 > 预定义默认值

    Args:
        config_path: 训练配置文件路径
        cli_overrides: 命令行覆盖参数
        defaults: 预定义默认值

    Returns:
        ConfigDict: 最终配置
    """
    # 1. 加载预定义默认值
    config = ConfigDict(defaults) if defaults else ConfigDict()

    # 2. 加载配置文件并合并
    if config_path:
        file_config = load_yaml_config(config_path)
        config = merge_configs(config, file_config)
        print(f"✓ 已加载训练配置: {config_path}")

    # 3. 应用命令行覆盖
    if cli_overrides:
        config = apply_cli_overrides(config, cli_overrides)

    return config


def load_robot_config(config_path: Union[str, Path], cli_overrides: Optional[Dict[str, Any]] = None) -> ConfigDict:
    """加载机器人配置

    Args:
        config_path: 机器人配置文件路径
        cli_overrides: 命令行覆盖参数

    Returns:
        ConfigDict: 机器人配置
    """
    config = load_yaml_config(config_path)
    print(f"✓ 已加载机器人配置: {config_path}")

    if cli_overrides:
        config = apply_cli_overrides(config, cli_overrides)

    return config


def load_servo_config(config_path: Union[str, Path], apply_calibration: bool = True) -> ConfigDict:
    """加载舵机配置（包含校准偏移）

    Args:
        config_path: 舵机配置文件路径
        apply_calibration: 是否应用校准偏移值

    Returns:
        ConfigDict: 舵机配置
    """
    config = load_yaml_config(config_path)
    print(f"✓ 已加载舵机配置: {config_path}")

    # 应用校准偏移
    if apply_calibration and "calibration" in config and "values" in config.calibration:
        print("✓ 应用舵机校准偏移:")
        calibration_values = config.calibration.values

        # 更新左脚踝舵机偏移
        if "left_ankle" in config:
            for servo_key in ["servo_1", "servo_2", "servo_3"]:
                if servo_key in config.left_ankle:
                    servo_id = config.left_ankle[servo_key]["id"]
                    if servo_id in calibration_values:
                        measured_offset = calibration_values[servo_id]["measured_offset"]
                        config.left_ankle[servo_key]["offset"] = measured_offset
                        print(f"  - 舵机 {servo_id} (左脚踝): offset = {measured_offset}")

        # 更新右脚踝舵机偏移
        if "right_ankle" in config:
            for servo_key in ["servo_1", "servo_2", "servo_3"]:
                if servo_key in config.right_ankle:
                    servo_id = config.right_ankle[servo_key]["id"]
                    if servo_id in calibration_values:
                        measured_offset = calibration_values[servo_id]["measured_offset"]
                        config.right_ankle[servo_key]["offset"] = measured_offset
                        print(f"  - 舵机 {servo_id} (右脚踝): offset = {measured_offset}")

    return config


def merge_configs(base: ConfigDict, override: ConfigDict) -> ConfigDict:
    """递归合并配置字典

    Args:
        base: 基础配置（低优先级）
        override: 覆盖配置（高优先级）

    Returns:
        ConfigDict: 合并后的配置
    """
    result = ConfigDict(base)

    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], ConfigDict(value))
        else:
            result[key] = value

    return result


def validate_train_config(config: ConfigDict) -> bool:
    """验证训练配置的完整性和合法性

    Args:
        config: 训练配置

    Returns:
        bool: 配置是否合法

    Raises:
        ValueError: 配置不合法
    """
    required_keys = ["task", "environment", "ppo"]

    # 检查必需键
    for key in required_keys:
        if key not in config:
            raise ValueError(f"训练配置缺少必需键: {key}")

    # 检查环境配置
    if "num_envs" not in config.environment:
        raise ValueError("环境配置缺少 num_envs")

    if config.environment.num_envs <= 0:
        raise ValueError(f"num_envs 必须 > 0，当前值: {config.environment.num_envs}")

    # 检查 PPO 配置
    if "algorithm" not in config.ppo or "runner" not in config.ppo:
        raise ValueError("PPO 配置缺少 algorithm 或 runner")

    # 检查学习率
    if "learning_rate" in config.ppo.algorithm:
        lr = config.ppo.algorithm.learning_rate
        if lr <= 0 or lr >= 1:
            raise ValueError(f"learning_rate 应在 (0, 1) 范围内，当前值: {lr}")

    print("✓ 训练配置验证通过")
    return True


def validate_robot_config(config: ConfigDict) -> bool:
    """验证机器人配置的完整性和合法性

    Args:
        config: 机器人配置

    Returns:
        bool: 配置是否合法

    Raises:
        ValueError: 配置不合法
    """
    required_keys = ["robot_info", "parallel_ankle", "action_mapping"]

    # 检查必需键
    for key in required_keys:
        if key not in config:
            raise ValueError(f"机器人配置缺少必需键: {key}")

    # 检查并联脚踝几何参数
    if "l0" not in config.parallel_ankle or "l1" not in config.parallel_ankle or "l2" not in config.parallel_ankle:
        raise ValueError("并联脚踝配置缺少几何参数 l0/l1/l2")

    # 检查动作空间映射
    if "action_dim" not in config.action_mapping:
        raise ValueError("动作空间配置缺少 action_dim")

    if "ankle_indices" not in config.action_mapping:
        raise ValueError("动作空间配置缺少 ankle_indices")

    ankle_indices = config.action_mapping.ankle_indices
    if "left" not in ankle_indices or "right" not in ankle_indices:
        raise ValueError("ankle_indices 必须包含 left 和 right")

    print("✓ 机器人配置验证通过")
    return True


def get_default_config_path(config_type: str) -> Path:
    """获取默认配置文件路径

    Args:
        config_type: 配置类型 ("train", "robot", "servo")

    Returns:
        Path: 配置文件路径
    """
    # 尝试从环境变量获取项目根目录
    project_root = os.environ.get("ISAACLAB_RL_ROOT")

    if project_root is None:
        # 如果环境变量不存在，使用相对路径
        current_file = Path(__file__).resolve()
        project_root = current_file.parents[3]  # isaaclab_rl/
    else:
        project_root = Path(project_root)

    config_dir = project_root / "configs"

    config_files = {
        "train": config_dir / "train_config.yaml",
        "robot": config_dir / "robot_config.yaml",
        "servo": config_dir / "servo_config.yaml",
    }

    if config_type not in config_files:
        raise ValueError(f"未知的配置类型: {config_type}。支持的类型: {list(config_files.keys())}")

    return config_files[config_type]


# 便捷函数，用于快速加载默认配置
def load_default_train_config(cli_overrides: Optional[Dict[str, Any]] = None) -> ConfigDict:
    """加载默认训练配置"""
    default_path = get_default_config_path("train")
    return load_train_config(default_path, cli_overrides=cli_overrides)


def load_default_robot_config(cli_overrides: Optional[Dict[str, Any]] = None) -> ConfigDict:
    """加载默认机器人配置"""
    default_path = get_default_config_path("robot")
    return load_robot_config(default_path, cli_overrides=cli_overrides)


def load_default_servo_config(apply_calibration: bool = True) -> ConfigDict:
    """加载默认舵机配置"""
    default_path = get_default_config_path("servo")
    return load_servo_config(default_path, apply_calibration=apply_calibration)
