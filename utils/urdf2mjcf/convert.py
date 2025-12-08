#!/usr/bin/env python3
"""
将 URDF 文件转换为 MJCF 格式
使用 urdf2mjcf 工具进行转换，并统一使用相对路径加载mesh文件
"""

import sys
import re
from pathlib import Path
from urdf2mjcf import run as convert_urdf_to_mjcf


def fix_mesh_paths(mjcf_path):
    """
    修复MJCF文件中的mesh路径，统一使用相对于MJCF文件的相对路径
    将 package://assets_jiyuan_right/meshes/ 替换为 ../meshes/

    Args:
        mjcf_path: MJCF文件路径
    """
    mjcf_path = Path(mjcf_path)

    if not mjcf_path.exists():
        print(f"警告: MJCF文件不存在: {mjcf_path}")
        return False

    # 读取文件内容
    content = mjcf_path.read_text(encoding='utf-8')

    original_content = content
    content = re.sub(
        r'package://[^/]+/meshes/',
        '../meshes/',
        content
    )

    # 如果内容有变化，写回文件
    if content != original_content:
        mjcf_path.write_text(content, encoding='utf-8')
        print("✓ 已将mesh路径统一为相对路径格式: ../meshes/")
        return True
    else:
        print("ℹ mesh路径已经是正确格式")
        return True


def fix_materials_and_visuals(mjcf_path):
    """
    修复MJCF文件中的材质和视觉属性，使其与jiyuan_backup.xml一致
    - 移除所有material定义（除了ground）
    - 移除default中的material引用
    - 直接在geom上使用rgba属性
    - 添加灯光和地板

    Args:
        mjcf_path: MJCF文件路径
    """
    mjcf_path = Path(mjcf_path)

    if not mjcf_path.exists():
        print(f"警告: MJCF文件不存在: {mjcf_path}")
        return False

    content = mjcf_path.read_text(encoding='utf-8')

    # 1. 移除整个asset部分的material定义（除了ground）
    # 匹配并删除所有 <material ... /> 行
    content = re.sub(
        r'\s*<material[^>]*?/>\s*\n',
        '',
        content
    )

    # 添加ground材质到asset中
    asset_match = re.search(r'(<asset>)', content)
    if asset_match:
        insert_pos = asset_match.end()
        ground_material = '\n        <material name="ground" rgba="0.8 0.8 0.8 1" />'
        content = content[:insert_pos] + ground_material + content[insert_pos:]

    # 2. 移除default中visual类的material属性
    # 将 material="visualgeom" 或 material="..." 移除
    content = re.sub(
        r'(<geom[^>]*?)\s+material="[^"]*"',
        r'\1',
        content
    )

    # 3. 为collision geom的default添加正确属性（如果还没有）
    # 确保collision类的default设置正确
    content = re.sub(
        r'(<default class="collision">\s*<geom[^>]*?)(/>)',
        r'\1 mass="0" \2',
        content
    )

    # 4. 为所有visual geom添加rgba属性
    # 先处理right_hip_roll_link - 特殊的浅蓝色
    content = re.sub(
        r'(<geom[^>]*?name="right_hip_roll_link"[^>]*?class="jiyuan"[^>]*?mesh="right_hip_roll_link"[^>]*?)(/>)',
        r'\1 rgba="0.792 0.820 0.933 1" \2',
        content
    )

    # 处理其他所有class="jiyuan"的geom - 白色
    # 避免重复添加rgba，先检查是否已有rgba
    lines = content.split('\n')
    new_lines = []
    for line in lines:
        if 'class="jiyuan"' in line and 'rgba=' not in line and 'right_hip_roll_link' not in line:
            # 在/> 之前插入rgba
            line = line.replace('/>', ' rgba="1 1 1 1" />')
        new_lines.append(line)
    content = '\n'.join(new_lines)

    # 5. 在worldbody中添加灯光和地板
    worldbody_match = re.search(r'(<worldbody>)', content)
    if worldbody_match:
        insert_pos = worldbody_match.end()
        lights_and_ground = '''
        <!-- 灯光 -->
        <light pos="0 0 3" dir="0 0 -1" diffuse="0.8 0.8 0.8" specular="0.2 0.2 0.2" directional="true" />

        <!-- 地板 -->
        <geom type="plane" size="10 10 0.1" pos="0 0 0" material="ground" condim="3" friction="1 0.005 0.0001" />
'''
        content = content[:insert_pos] + \
            lights_and_ground + content[insert_pos:]

    # 6. 在mujoco根标签后添加visual质量设置
    mujoco_match = re.search(r'(<mujoco[^>]*?>)', content)
    if mujoco_match:
        insert_pos = mujoco_match.end()
        visual_section = '''
    <visual>
        <quality shadowsize="4096" />
    </visual>
'''
        content = content[:insert_pos] + visual_section + content[insert_pos:]

    mjcf_path.write_text(content, encoding='utf-8')
    print("✓ 已修复材质、视觉属性，并添加灯光、地板和视觉质量设置")
    return True


def convert_urdf_to_mjcf_wrapper(urdf_path, output_path):
    """
    使用 urdf2mjcf 库转换 URDF 到 MJCF，并统一文件路径

    Args:
        urdf_path: URDF 文件路径
        output_path: 输出 MJCF 文件路径
    """
    urdf_path = Path(urdf_path).resolve()
    output_path = Path(output_path).resolve()

    print(f"转换 URDF 文件: {urdf_path}")
    print(f"输出 MJCF 文件: {output_path}")

    # 创建输出目录
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 运行转换
    try:
        convert_urdf_to_mjcf(
            urdf_path=str(urdf_path),
            mjcf_path=str(output_path),
            copy_meshes=False,
        )
        print("✓ URDF到MJCF转换成功！")

        # 修复mesh路径
        print("\n修复mesh文件路径...")
        if not fix_mesh_paths(output_path):
            print("✗ 路径修复失败")
            return False

        # 修复材质和视觉属性
        print("\n修复材质和视觉属性...")
        if not fix_materials_and_visuals(output_path):
            print("✗ 材质修复失败")
            return False

        print(f"\n✓ 完成！生成的 MJCF 文件: {output_path}")
        return True

    except Exception as e:
        print(f"✗ 转换失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # URDF 文件路径
    urdf_path = Path(__file__).parent.parent / \
        "assets_jiyuan_right" / "urdf" / "assets_jiyuan_right.urdf"

    # 输出 MJCF 文件路径
    output_path = Path(__file__).parent.parent / \
        "assets_jiyuan_right" / "mjcf" / "jiyuan_converted.xml"

    success = convert_urdf_to_mjcf_wrapper(urdf_path, output_path)
    sys.exit(0 if success else 1)
