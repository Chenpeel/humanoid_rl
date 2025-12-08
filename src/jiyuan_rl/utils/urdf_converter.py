"""
URDF到MuJoCo XML转换工具
"""

import os
from pathlib import Path
import xml.etree.ElementTree as ET
from typing import Optional
from rich.console import Console

console = Console()


def urdf_to_mjcf(
    urdf_path: str,
    output_path: Optional[str] = None,
    mesh_dir: Optional[str] = None,
) -> str:
    """
    将URDF文件转换为MuJoCo MJCF格式

    Args:
        urdf_path: URDF文件路径
        output_path: 输出MJCF路径（如果为None，自动生成）
        mesh_dir: 网格文件目录（如果为None，使用URDF相对路径）

    Returns:
        输出MJCF文件路径
    """
    import mujoco
    import re

    urdf_path = Path(urdf_path).resolve()

    if output_path is None:
        output_path = urdf_path.with_suffix('.xml')
    else:
        output_path = Path(output_path).resolve()

    console.print(f"[cyan]转换URDF到MJCF[/cyan]")
    console.print(f"  输入: {urdf_path}")
    console.print(f"  输出: {output_path}")

    # 读取URDF内容
    with open(urdf_path, 'r', encoding='utf-8') as f:
        urdf_content = f.read()

    # 修复mesh路径：package://assets/meshes/xxx.STL -> ../meshes/xxx.STL
    # 或者使用绝对路径
    if mesh_dir is None:
        # 默认mesh_dir在urdf的上级目录
        mesh_dir = urdf_path.parent.parent / "meshes"
    else:
        mesh_dir = Path(mesh_dir)

    console.print(f"  网格目录: {mesh_dir}")

    # 替换package://路径为绝对路径
    def replace_mesh_path(match):
        filename = match.group(1)
        # 使用绝对路径
        abs_path = (mesh_dir / filename).resolve()
        return f'filename="{abs_path}"'

    # 替换所有package://路径（支持多种格式）
    # 匹配 package://xxx/meshes/file.STL 或 package://xxx/file.STL
    urdf_content_fixed = re.sub(
        r'filename="package://[^/]+(?:/meshes)?/([^"]+)"',
        replace_mesh_path,
        urdf_content
    )

    # 保存临时URDF
    temp_urdf = output_path.parent / f"temp_{urdf_path.name}"
    with open(temp_urdf, 'w', encoding='utf-8') as f:
        f.write(urdf_content_fixed)

    console.print(f"  [dim]临时URDF: {temp_urdf}[/dim]")

    # 检查第一个mesh路径
    first_mesh = re.search(
        r'filename="([^"]+\.STL)"', urdf_content_fixed, re.IGNORECASE)
    if first_mesh:
        console.print(f"  [dim]示例mesh路径: {first_mesh.group(1)}[/dim]")
        if not Path(first_mesh.group(1)).exists():
            console.print(f"  [red]警告: mesh文件不存在！[/red]")

    try:
        # 使用MuJoCo的URDF加载器
        model = mujoco.MjModel.from_xml_path(str(temp_urdf))

        # 保存为XML
        mujoco.mj_saveLastXML(str(output_path), model)

        console.print(f"  [green]✓ 转换成功！[/green]")
        return str(output_path)

    except Exception as e:
        console.print(f"  [red]✗ 转换失败: {e}[/red]")
        raise
    finally:
        # 删除临时文件
        if temp_urdf.exists():
            temp_urdf.unlink()


def create_jiyuan_scene_xml(
    robot_mjcf: str,
    output_path: str,
    ground_size: tuple = (20, 20, 0.1),
    ground_friction: tuple = (1.0, 0.005, 0.0001),
    robot_pos: tuple = (0, 0, 0.3),
) -> str:
    """
    创建包含机器人的完整场景XML

    Args:
        robot_mjcf: 机器人MJCF文件路径（已转换的XML）
        output_path: 输出XML路径
        ground_size: 地面大小 (长, 宽, 高)
        ground_friction: 地面摩擦力参数
        robot_pos: 机器人初始位置

    Returns:
        输出XML文件路径
    """
    console.print(f"[cyan]创建场景XML[/cyan]")

    # 创建根元素
    mujoco = ET.Element('mujoco', model='jiyuan_scene')

    # 编译器设置
    compiler = ET.SubElement(mujoco, 'compiler', {
        'angle': 'radian',
        'coordinate': 'local',
        'inertiafromgeom': 'true',
    })

    # 选项设置
    option = ET.SubElement(mujoco, 'option', {
        'timestep': '0.002',
        'iterations': '50',
        'solver': 'Newton',
        'gravity': '0 0 -9.81',
    })

    # 资源
    asset = ET.SubElement(mujoco, 'asset')
    ET.SubElement(asset, 'texture', {
        'name': 'grid',
        'type': '2d',
        'builtin': 'checker',
        'width': '512',
        'height': '512',
        'rgb1': '0.1 0.2 0.3',
        'rgb2': '0.2 0.3 0.4',
    })
    ET.SubElement(asset, 'material', {
        'name': 'grid',
        'texture': 'grid',
        'texrepeat': '1 1',
        'texuniform': 'true',
        'reflectance': '0.2',
    })

    # 世界体
    worldbody = ET.SubElement(mujoco, 'worldbody')

    # 光源
    ET.SubElement(worldbody, 'light', {
        'name': 'spotlight',
        'mode': 'targetbodycom',
        'target': 'base_link',
        'diffuse': '0.8 0.8 0.8',
        'specular': '0.3 0.3 0.3',
        'pos': '0 -6 4',
        'dir': '0 1 -1',
    })

    # 地面
    floor = ET.SubElement(worldbody, 'geom', {
        'name': 'floor',
        'type': 'plane',
        'size': f'{ground_size[0]} {ground_size[1]} {ground_size[2]}',
        'material': 'grid',
        'friction': f'{ground_friction[0]} {ground_friction[1]} {ground_friction[2]}',
        'condim': '3',
    })

    # 包含机器人MJCF（已转换）
    robot_mjcf_path = Path(robot_mjcf).resolve()
    ET.SubElement(mujoco, 'include', {
        'file': str(robot_mjcf_path),
    })

    # 保存XML
    tree = ET.ElementTree(mujoco)
    ET.indent(tree, space="  ")
    tree.write(output_path, encoding='utf-8', xml_declaration=True)

    console.print(f"  [green]✓ 场景XML创建成功: {output_path}[/green]")
    return output_path


def setup_jiyuan_urdf(
    urdf_path: str = None,
    output_dir: str = None,
) -> str:
    """
    设置机器人URDF环境

    Args:
        urdf_path: URDF文件路径（如果为None，使用assets/urdf/assets.urdf）
        output_dir: 输出目录（如果为None，使用assets/scenes/）

    Returns:
        场景XML文件路径
    """
    if urdf_path is None:
        # 默认使用assets_jiyuan_right中的URDF
        script_dir = Path(__file__).parent.parent.parent.parent  # 到rl/目录
        urdf_path = script_dir / "assets_jiyuan_right" / \
            "urdf" / "assets_jiyuan_right.urdf"

    urdf_path = Path(urdf_path).resolve()

    if not urdf_path.exists():
        raise FileNotFoundError(f"找不到URDF文件: {urdf_path}")

    if output_dir is None:
        output_dir = urdf_path.parent.parent / "scenes"
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    console.print(f"[cyan]设置机器人环境[/cyan]")
    console.print(f"  URDF: {urdf_path}")
    console.print(f"  输出目录: {output_dir}")

    # 步骤1: 转换URDF到MJCF
    robot_mjcf = output_dir / "jiyuan_robot.xml"
    console.print(f"\n[cyan]步骤1: 转换URDF到MJCF[/cyan]")
    try:
        urdf_to_mjcf(str(urdf_path), str(robot_mjcf))
    except Exception as e:
        console.print(f"[red]✗ URDF转换失败: {e}[/red]")
        console.print(f"[yellow]提示: URDF文件可能包含MuJoCo不支持的元素[/yellow]")
        raise

    # 步骤2: 创建完整场景
    scene_xml = output_dir / "jiyuan_scene.xml"
    console.print(f"\n[cyan]步骤2: 创建场景XML[/cyan]")
    scene_xml_path = create_jiyuan_scene_xml(
        robot_mjcf=str(robot_mjcf),
        output_path=str(scene_xml),
    )

    console.print(f"\n[green]✓ 环境设置完成: {scene_xml_path}[/green]")
    return scene_xml_path
