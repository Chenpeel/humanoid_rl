#!/usr/bin/env python3
"""
URDF转MJCF工具，支持旋转和抬高模型

主要功能：
1. 将单腿MJCF通过YZ镜像自动生成双腿MJCF
2. 旋转模型并抬高base_link
3. 完整的URDF到双腿MJCF转换流程
"""

import argparse
import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


# ============================================================================================
# ======================================= 模块导入 ============================================
# ============================================================================================

def import_convert_module():
    try:
        from .convert import convert_urdf_to_mjcf_wrapper
        return convert_urdf_to_mjcf_wrapper
    except ImportError:
        current_dir = Path(__file__).parent
        if str(current_dir) not in sys.path:
            sys.path.insert(0, str(current_dir))
        from convert import convert_urdf_to_mjcf_wrapper
        return convert_urdf_to_mjcf_wrapper

# --------------------------------------------------------------------------------------------

def import_mirror_module():
    try:
        from .mirror_mjcf import mirror_mjcf
        return mirror_mjcf
    except ImportError:
        current_dir = Path(__file__).parent
        if str(current_dir) not in sys.path:
            sys.path.insert(0, str(current_dir))
        from mirror_mjcf import mirror_mjcf
        return mirror_mjcf

convert_urdf_to_mjcf_wrapper = None
mirror_mjcf = None

# ============================================================================================
# ===================================== END: 模块导入 ==========================================
# ============================================================================================


# ============================================================================================
# ======================================= 旋转与抬高 ===========================================
# ============================================================================================

def rotate_and_elevate_mjcf(
    mjcf_path, output_path=None, rotation_angle=-90, elevation_height=1.1
):
    """旋转MJCF模型并抬高base_link"""
    try:
        mjcf_path = Path(mjcf_path)
        if output_path is None:
            output_path = mjcf_path
        else:
            output_path = Path(output_path)

        tree = ET.parse(mjcf_path)
        root = tree.getroot()

        worldbody = root.find("worldbody")
        if worldbody is None:
            print("错误: 找不到worldbody元素")
            return False

        base_link = None
        for body in worldbody.findall("body"):
            if body.attrib.get("name", "") == "base_link":
                base_link = body
                break

        if base_link is None:
            print("错误: 找不到base_link")
            return False

        angle_rad = math.radians(rotation_angle)
        w = math.cos(angle_rad / 2)
        x = math.sin(angle_rad / 2)
        y = 0
        z = 0
        rotation_quat = f"{w:.8f} {x:.8f} {y:.8f} {z:.8f}"

        current_pos = base_link.attrib.get("pos", "0 0 0")
        pos_parts = current_pos.split()
        if len(pos_parts) >= 3:
            try:
                x_pos = float(pos_parts[0])
                y_pos = float(pos_parts[1])
                z_pos = float(pos_parts[2])
                z_pos += elevation_height

                new_pos = f"{x_pos:.8f} {y_pos:.8f} {z_pos:.8f}"
                base_link.attrib["pos"] = new_pos
                base_link.attrib["quat"] = rotation_quat

                print(f"✓ 已将模型绕X轴旋转 {rotation_angle} 度并抬高到 {z_pos:.3f} 米")
            except ValueError:
                print("错误: 无法解析base_link的位置坐标")
                return False
        else:
            new_pos = f"0.0 0.0 {elevation_height:.8f}"
            base_link.attrib["pos"] = new_pos
            base_link.attrib["quat"] = rotation_quat
            print(f"✓ 已添加绕X轴旋转 {rotation_angle} 度并设置高度为 {elevation_height:.3f} 米")

        tree.write(output_path, encoding="utf-8", xml_declaration=True)
        print(f"✓ 已保存修改后的MJCF文件: {output_path}")
        return True

    except Exception as e:
        print(f"✗ 旋转和抬高操作失败: {e}")
        import traceback
        traceback.print_exc()
        return False

# ============================================================================================
# ===================================== END: 旋转与抬高 ========================================
# ============================================================================================


# ============================================================================================
# ======================================= 管道逻辑 ============================================
# ============================================================================================

def urdf2mjcf_full_pipeline(
    urdf_path, mjcf_right_path, mjcf_full_path, rotation_angle=-90, elevation_height=1.1
):
    """URDF转MJCF完整流程，包含旋转和抬高"""
    global convert_urdf_to_mjcf_wrapper, mirror_mjcf
    if convert_urdf_to_mjcf_wrapper is None:
        convert_urdf_to_mjcf_wrapper = import_convert_module()
    if mirror_mjcf is None:
        mirror_mjcf = import_mirror_module()

    print("[1/2] URDF转MJCF（右腿）...")
    ok = convert_urdf_to_mjcf_wrapper(urdf_path, mjcf_right_path)
    if not ok:
        print("✗ URDF转MJCF失败")
        return False

    print("[2/2] 镜像生成双腿...")
    mirror_mjcf(mjcf_right_path, mjcf_full_path)

    print("[3/3] 旋转并抬高模型...")
    rotate_and_elevate_mjcf(
        mjcf_full_path, mjcf_full_path, rotation_angle, elevation_height
    )

    print(f"✓ 已生成完整双腿MJCF: {mjcf_full_path}")
    return True

# ============================================================================================
# ===================================== END: 管道逻辑 ==========================================
# ============================================================================================


# ============================================================================================
# ======================================= 主函数 ==============================================
# ============================================================================================

def main():
    import os

    parser = argparse.ArgumentParser(
        description="URDF转双腿MJCF自动化工具，支持旋转和抬高模型",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--urdf", type=str, help="输入URDF文件路径（传统方式）")
    parser.add_argument("--mjcf", type=str, help="输出双腿MJCF文件路径（传统方式）")
    parser.add_argument("--tmp", type=str, default=None, help="中间右腿MJCF文件路径")
    parser.add_argument("--rotate", type=float, default=-90.0, help="绕X轴旋转角度")
    parser.add_argument("--elevate", type=float, default=1.1, help="抬高高度")

    subparsers = parser.add_subparsers(dest="command", help="选择操作模式")

    convert_parser = subparsers.add_parser("convert", help="完整URDF转MJCF流程")
    convert_parser.add_argument("--urdf", type=str, required=True)
    convert_parser.add_argument("--mjcf", type=str, required=True)
    convert_parser.add_argument("--tmp", type=str, default=None)
    convert_parser.add_argument("--rotate", type=float, default=-90.0)
    convert_parser.add_argument("--elevate", type=float, default=1.1)

    rotate_parser = subparsers.add_parser("rotate", help="仅旋转和抬高现有MJCF文件")
    rotate_parser.add_argument("--input", type=str, required=True)
    rotate_parser.add_argument("--output", type=str, default=None)
    rotate_parser.add_argument("--rotate", type=float, default=-90.0)
    rotate_parser.add_argument("--elevate", type=float, default=1.1)

    args = parser.parse_args()

    if args.command is None:
        if args.urdf is None or args.mjcf is None:
            parser.print_help()
            sys.exit(1)

        urdf_path = Path(args.urdf).resolve()
        mjcf_full_path = Path(args.mjcf).resolve()
        user_tmp = args.tmp is not None
        if user_tmp:
            mjcf_right_path = Path(args.tmp).resolve()
        else:
            mjcf_right_path = mjcf_full_path.parent / (mjcf_full_path.stem + "_right.xml")

        urdf2mjcf_full_pipeline(
            urdf_path,
            mjcf_right_path,
            mjcf_full_path,
            rotation_angle=args.rotate,
            elevation_height=args.elevate,
        )

        if not user_tmp and mjcf_right_path.exists():
            try:
                os.remove(mjcf_right_path)
            except Exception:
                pass

    elif args.command == "convert":
        urdf_path = Path(args.urdf).resolve()
        mjcf_full_path = Path(args.mjcf).resolve()
        user_tmp = args.tmp is not None
        if user_tmp:
            mjcf_right_path = Path(args.tmp).resolve()
        else:
            mjcf_right_path = mjcf_full_path.parent / (mjcf_full_path.stem + "_right.xml")

        urdf2mjcf_full_pipeline(
            urdf_path,
            mjcf_right_path,
            mjcf_full_path,
            rotation_angle=args.rotate,
            elevation_height=args.elevate,
        )

        if not user_tmp and mjcf_right_path.exists():
            try:
                os.remove(mjcf_right_path)
            except Exception:
                pass

    elif args.command == "rotate":
        input_path = Path(args.input).resolve()
        if not input_path.exists():
            print(f"错误: 输入文件不存在: {input_path}")
            sys.exit(1)

        output_path = args.output
        if output_path is None:
            output_path = input_path
        else:
            output_path = Path(output_path).resolve()

        success = rotate_and_elevate_mjcf(
            input_path,
            output_path,
            rotation_angle=args.rotate,
            elevation_height=args.elevate,
        )

        if not success:
            sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("URDF转MJCF工具，支持旋转和抬高模型")
        sys.exit(0)
    main()

# ============================================================================================
# ===================================== END: 主函数 ============================================
# ============================================================================================