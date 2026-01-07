"""
机器人模型配置管理器

支持多机器人模型的统一配置和路径管理，提供扩展性和向后兼容性。
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import jax.numpy as jp


@dataclass
class RobotConfig:
    """机器人配置数据类

    Attributes:
        name: 机器人名称（唯一标识）
        model_dir: 模型目录路径（相对于项目根目录）
        model_file: 主模型XML文件名
        scene_file: 场景XML文件名
        base_rotation: 基座坐标系旋转四元数（用于坐标系对齐）
        description: 机器人描述
    """
    name: str
    model_dir: str
    model_file: str
    scene_file: str
    base_rotation: Optional[Tuple[float, float, float, float]] = None
    nominal_height: float = 1.0  # 机器人的标准站立高度
    description: str = ""

    def get_model_path(self, project_root: Optional[Path] = None) -> Path:
        """获取模型XML的完整路径

        Args:
            project_root: 项目根目录，如果为None则自动推断

        Returns:
            模型XML文件的完整路径
        """
        if project_root is None:
            project_root = Path(__file__).parent.parent.parent.parent
        return project_root / self.model_dir / self.model_file

    def get_scene_path(self, project_root: Optional[Path] = None) -> Path:
        """获取场景XML的完整路径

        Args:
            project_root: 项目根目录，如果为None则自动推断

        Returns:
            场景XML文件的完整路径
        """
        if project_root is None:
            project_root = Path(__file__).parent.parent.parent.parent
        return project_root / self.model_dir / self.scene_file

    def get_base_rotation_array(self) -> Optional[jp.ndarray]:
        """获取基座旋转四元数的JAX数组表示

        Returns:
            基座旋转四元数（w, x, y, z）或None
        """
        if self.base_rotation is None:
            return None
        return jp.array(self.base_rotation)


# ============================================================================================
# ======================================= 机器人配置注册表 =====================================
# ============================================================================================

# 已注册的机器人配置
ROBOT_CONFIGS: Dict[str, RobotConfig] = {
    # Gaoda Jiyuan 机器人
    "gaoda_jiyuan": RobotConfig(
        name="gaoda_jiyuan",
        model_dir="robots/gaoda_jiyuan",
        model_file="jiyuan.xml",
        scene_file="scene.xml",
        base_rotation=(0.70710678, 0.70710678, 0.0, 0.0),  # 90度绕X轴旋转（Y-up → Z-up）
        nominal_height=1.00,
        description="Gaoda Jiyuan 双足机器人（原始模型为Y-up，已校正为Z-up）"
    ),

    # Unitree H1 机器人
    "unitree_h1": RobotConfig(
        name="unitree_h1",
        model_dir="robots/unitree_h1",
        model_file="h1.xml",
        scene_file="scene.xml",
        base_rotation=None,  # H1模型已经是Z-up坐标系，无需校正
        nominal_height=0.98,
        description="Unitree H1 人形机器人"
    ),

    # 向后兼容：旧的路径配置（映射到gaoda_jiyuan）
    "jiyuan_legacy": RobotConfig(
        name="jiyuan_legacy",
        model_dir="assets/xmls",
        model_file="models/jiyuan_fit.xml",
        scene_file="scenes/flat_terrain.xml",
        base_rotation=(0.70710678, 0.70710678, 0.0, 0.0),
        nominal_height=0.78,
        description="[已弃用] 旧的Jiyuan模型路径配置，建议迁移到gaoda_jiyuan"
    ),
}

# 默认机器人配置
DEFAULT_ROBOT = "gaoda_jiyuan"


# ============================================================================================
# ======================================= 辅助函数 ============================================
# ============================================================================================


def get_robot_config(robot_name: Optional[str] = None) -> RobotConfig:
    """获取机器人配置

    Args:
        robot_name: 机器人名称，如果为None则使用默认机器人

    Returns:
        机器人配置对象

    Raises:
        ValueError: 如果指定的机器人名称不存在
    """
    if robot_name is None:
        robot_name = DEFAULT_ROBOT

    if robot_name not in ROBOT_CONFIGS:
        available = ", ".join(ROBOT_CONFIGS.keys())
        raise ValueError(
            f"未找到机器人配置: {robot_name}\n"
            f"可用的机器人: {available}"
        )

    return ROBOT_CONFIGS[robot_name]


def register_robot(config: RobotConfig) -> None:
    """注册新的机器人配置

    Args:
        config: 机器人配置对象

    Raises:
        ValueError: 如果机器人名称已存在
    """
    if config.name in ROBOT_CONFIGS:
        raise ValueError(f"机器人配置已存在: {config.name}")

    ROBOT_CONFIGS[config.name] = config


def list_available_robots() -> Dict[str, str]:
    """列出所有可用的机器人配置

    Returns:
        字典，键为机器人名称，值为描述
    """
    return {name: cfg.description for name, cfg in ROBOT_CONFIGS.items()}


def resolve_scene_path(
    robot_name: Optional[str] = None,
    scene_file: Optional[str] = None,
    project_root: Optional[Path] = None
) -> Path:
    """解析场景XML路径（统一路径解析接口）

    Args:
        robot_name: 机器人名称
        scene_file: 场景文件名（可选，覆盖默认场景）
        project_root: 项目根目录

    Returns:
        场景XML文件的完整路径
    """
    config = get_robot_config(robot_name)

    if scene_file is not None:
        # 如果指定了自定义场景文件，使用模型目录 + 自定义场景
        if project_root is None:
            project_root = Path(__file__).parent.parent.parent.parent
        return project_root / config.model_dir / scene_file
    else:
        # 否则使用配置中的默认场景
        return config.get_scene_path(project_root)


# ============================================================================================
# ======================================= 调试输出 ============================================
# ============================================================================================


def print_robot_configs():
    """打印所有注册的机器人配置（用于调试）"""
    from rich.console import Console
    from rich.table import Table

    console = Console()
    table = Table(title="已注册的机器人配置")
    table.add_column("名称", style="cyan")
    table.add_column("模型目录", style="green")
    table.add_column("模型文件", style="yellow")
    table.add_column("场景文件", style="yellow")
    table.add_column("坐标系校正", style="magenta")
    table.add_column("描述", style="white")

    for name, cfg in ROBOT_CONFIGS.items():
        rotation_str = "是" if cfg.base_rotation is not None else "否"
        table.add_row(
            name,
            cfg.model_dir,
            cfg.model_file,
            cfg.scene_file,
            rotation_str,
            cfg.description
        )

    console.print(table)


if __name__ == "__main__":
    # 测试代码
    print_robot_configs()

    # 测试路径解析
    config = get_robot_config("gaoda_jiyuan")
    print(f"\n模型路径: {config.get_model_path()}")
    print(f"场景路径: {config.get_scene_path()}")
    print(f"基座旋转: {config.get_base_rotation_array()}")
