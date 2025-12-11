"""
URDF到MJCF转换工具
用于将URDF格式的机器人模型转换为MuJoCo的MJCF格式
"""

import os
from pathlib import Path
from typing import Optional


def urdf_to_mjcf(
    urdf_path: str,
    output_path: Optional[str] = None,
    meshdir: Optional[str] = None,
) -> str:
    """将URDF文件转换为MJCF格式

    Args:
        urdf_path: URDF文件路径
        output_path: 输出的MJCF文件路径，如果为None则自动生成
        meshdir: mesh文件目录路径

    Returns:
        生成的MJCF文件路径
    """
    try:
        import mujoco
    except ImportError:
        raise ImportError("需要安装mujoco包: pip install mujoco")

    # 读取URDF文件
    with open(urdf_path, 'r') as f:
        urdf_content = f.read()

    # 如果未指定输出路径，自动生成
    if output_path is None:
        urdf_path_obj = Path(urdf_path)
        output_path = str(urdf_path_obj.parent / f"{urdf_path_obj.stem}.xml")

    # MuJoCo会自动处理URDF到MJCF的转换
    # 这里我们直接保存URDF路径，让MuJoCo在加载时转换
    # 或者使用mujoco.MjModel.from_xml_string进行转换

    try:
        # 加载URDF并保存为MJCF
        model = mujoco.MjModel.from_xml_path(urdf_path)
        mujoco.mj_saveLastXML(output_path, model)
        return output_path
    except Exception as e:
        raise RuntimeError(f"URDF转换失败: {e}")


def create_jiyuan_scene_xml(
    robot_xml_path: str,
    output_path: str = "scene.xml",
    floor_size: tuple = (10, 10, 0.1),
) -> str:
    """创建包含机器人和环境的场景XML文件

    Args:
        robot_xml_path: 机器人XML文件路径
        output_path: 输出的场景XML文件路径
        floor_size: 地板大小 (x, y, z)

    Returns:
        生成的场景XML文件路径
    """
    scene_template = f"""
<mujoco model="jiyuan_scene">
    <compiler angle="radian" meshdir="meshes"/>

    <option timestep="0.002" gravity="0 0 -9.81">
        <flag warmstart="enable"/>
    </option>

    <visual>
        <headlight ambient="0.5 0.5 0.5"/>
    </visual>

    <asset>
        <texture name="grid" type="2d" builtin="checker" width="512" height="512"
                 rgb1="0.1 0.1 0.1" rgb2="0.2 0.2 0.2"/>
        <material name="grid" texture="grid" texrepeat="1 1" texuniform="true"
                  reflectance="0.2"/>
    </asset>

    <worldbody>
        <!-- Ground plane -->
        <geom name="floor" type="plane" size="{floor_size[0]} {floor_size[1]} {floor_size[2]}"
              material="grid" condim="3" friction="1 0.005 0.0001"/>

        <!-- Lighting -->
        <light name="top" pos="0 0 3" dir="0 0 -1" directional="true"/>

        <!-- Include robot -->
        <include file="{robot_xml_path}"/>
    </worldbody>

    <actuator>
        <!-- Actuators will be defined by the included robot model -->
    </actuator>
</mujoco>
"""

    # 写入文件
    with open(output_path, 'w') as f:
        f.write(scene_template.strip())

    return output_path


def setup_jiyuan_urdf(
    urdf_path: str,
    assets_dir: str = "assets",
    create_scene: bool = True,
) -> dict:
    """设置机器人URDF的完整工作流程

    Args:
        urdf_path: URDF文件路径
        assets_dir: 资源目录路径
        create_scene: 是否创建场景文件

    Returns:
        包含文件路径的字典
    """
    assets_path = Path(assets_dir)
    assets_path.mkdir(exist_ok=True)

    # 转换URDF到MJCF
    xml_dir = assets_path / "xmls"
    xml_dir.mkdir(exist_ok=True)

    robot_xml = str(xml_dir / "robot.xml")
    urdf_to_mjcf(urdf_path, robot_xml)

    result = {
        "robot_xml": robot_xml,
    }

    # 创建场景
    if create_scene:
        scene_xml = str(xml_dir / "scene.xml")
        create_jiyuan_scene_xml(robot_xml, scene_xml)
        result["scene_xml"] = scene_xml

    return result


# 便捷函数
def quick_convert(urdf_path: str) -> str:
    """快速转换URDF到MJCF

    Args:
        urdf_path: URDF文件路径

    Returns:
        转换后的MJCF文件路径
    """
    return urdf_to_mjcf(urdf_path)
