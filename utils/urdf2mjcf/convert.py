#!/usr/bin/env python3
"""
将 URDF 文件转换为 MJCF 格式
使用 urdf2mjcf 工具进行转换，并统一使用相对路径加载mesh文件
"""

import sys
import re
from pathlib import Path
from urdf2mjcf import run as convert_urdf_to_mjcf


# ============================================================================================
# ======================================= 路径与材质修复 =======================================
# ============================================================================================

def fix_mesh_paths(mjcf_path):
    """修复MJCF文件中的mesh路径"""
    mjcf_path = Path(mjcf_path)

    if not mjcf_path.exists():
        print(f"警告: MJCF文件不存在: {mjcf_path}")
        return False

    content = mjcf_path.read_text(encoding='utf-8')
    original_content = content
    content = re.sub(
        r'package://[^/]+/meshes/',
        '../meshes/',
        content
    )

    if content != original_content:
        mjcf_path.write_text(content, encoding='utf-8')
        print("✓ 已将mesh路径统一为相对路径格式: ../meshes/")
        return True
    else:
        print("ℹ mesh路径已经是正确格式")
        return True

# --------------------------------------------------------------------------------------------

def fix_materials_and_visuals(mjcf_path):
    """修复MJCF文件中的材质和视觉属性"""
    mjcf_path = Path(mjcf_path)

    if not mjcf_path.exists():
        print(f"警告: MJCF文件不存在: {mjcf_path}")
        return False

    content = mjcf_path.read_text(encoding='utf-8')

    # 1. 移除 material 定义
    content = re.sub(r'\s*<material[^>]*?/>\s*\n', '', content)

    # 添加 ground 材质
    asset_match = re.search(r'(<asset>)', content)
    if asset_match:
        insert_pos = asset_match.end()
        ground_material = '\n        <material name="ground" rgba="0.8 0.8 0.8 1" />'
        content = content[:insert_pos] + ground_material + content[insert_pos:]

    # 2. 移除 default 中的 material 引用
    content = re.sub(r'(<geom[^>]*?)\s+material="[^"]*"', r'\1', content)

    # 3. 修复 collision geom 的 default
    content = re.sub(r'(<default class="collision">\s*<geom[^>]*?)(/>)', r'\1 mass="0" \2', content)

    # 4. 为 visual geom 添加 rgba
    content = re.sub(
        r'(<geom[^>]*?name="right_hip_roll_link"[^>]*?class="jiyuan"[^>]*?mesh="right_hip_roll_link"[^>]*?)(/>)',
        r'\1 rgba="0.792 0.820 0.933 1" \2',
        content
    )

    lines = content.split('\n')
    new_lines = []
    for line in lines:
        if 'class="jiyuan"' in line and 'rgba=' not in line and 'right_hip_roll_link' not in line:
            line = line.replace('/>', ' rgba="1 1 1 1" />')
        new_lines.append(line)
    content = '\n'.join(new_lines)

    # 5. 添加灯光和地板
    worldbody_match = re.search(r'(<worldbody>)', content)
    if worldbody_match:
        insert_pos = worldbody_match.end()
        lights_and_ground = '''
        <!-- 灯光 -->
        <light pos="0 0 3" dir="0 0 -1" diffuse="0.8 0.8 0.8" specular="0.2 0.2 0.2" directional="true" />

        <!-- 地板 -->
        <geom type="plane" size="10 10 0.1" pos="0 0 0" material="ground" condim="3" friction="1 0.005 0.0001" />
'''
        content = content[:insert_pos] + lights_and_ground + content[insert_pos:]

    # 6. 添加 visual 质量设置
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

# ============================================================================================
# ===================================== END: 路径与材质修复 ====================================
# ============================================================================================


# ============================================================================================
# ======================================= 转换包装器 ==========================================
# ============================================================================================

def convert_urdf_to_mjcf_wrapper(urdf_path, output_path):
    """使用 urdf2mjcf 库转换 URDF 到 MJCF"""
    urdf_path = Path(urdf_path).resolve()
    output_path = Path(output_path).resolve()

    print(f"转换 URDF 文件: {urdf_path}")
    print(f"输出 MJCF 文件: {output_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        convert_urdf_to_mjcf(
            urdf_path=str(urdf_path),
            mjcf_path=str(output_path),
            copy_meshes=False,
        )
        print("✓ URDF到MJCF转换成功！")

        print("\n修复mesh文件路径...")
        if not fix_mesh_paths(output_path):
            print("✗ 路径修复失败")
            return False

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

# ============================================================================================
# ===================================== END: 转换包装器 ========================================
# ============================================================================================


# ============================================================================================
# ======================================= 主函数 ==============================================
# ============================================================================================

if __name__ == "__main__":
    urdf_path = Path(__file__).parent.parent /
        "assets_jiyuan_right" / "urdf" / "assets_jiyuan_right.urdf"

    output_path = Path(__file__).parent.parent /
        "assets_jiyuan_right" / "mjcf" / "jiyuan_converted.xml"

    success = convert_urdf_to_mjcf_wrapper(urdf_path, output_path)
    sys.exit(0 if success else 1)

# ============================================================================================
# ===================================== END: 主函数 ============================================
# ============================================================================================