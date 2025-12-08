#!/usr/bin/env python3
"""
URDF转MJCF工具，支持旋转和抬高模型

主要功能：
1. 将单腿MJCF通过YZ镜像自动生成双腿MJCF
2. 旋转模型并抬高base_link
3. 完整的URDF到双腿MJCF转换流程

使用示例：
1. 完整转换流程：
   python urdf2mjcf.py --urdf input.urdf --mjcf output.xml --rotate -90 --elevate 1.1

2. 仅旋转和抬高现有MJCF文件：
   python urdf2mjcf.py --rotate-mjcf input.xml --output rotated.xml --rotate -90 --elevate 1.1
"""

import argparse
import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


# 动态导入以避免循环导入
def import_convert_module():
    try:
        from .convert import convert_urdf_to_mjcf_wrapper

        return convert_urdf_to_mjcf_wrapper
    except ImportError:
        # 添加到系统路径
        current_dir = Path(__file__).parent
        if str(current_dir) not in sys.path:
            sys.path.insert(0, str(current_dir))
        from convert import convert_urdf_to_mjcf_wrapper

        return convert_urdf_to_mjcf_wrapper


def import_mirror_module():
    try:
        from .mirror_mjcf import mirror_mjcf

        return mirror_mjcf
    except ImportError:
        # 添加到系统路径
        current_dir = Path(__file__).parent
        if str(current_dir) not in sys.path:
            sys.path.insert(0, str(current_dir))
        from mirror_mjcf import mirror_mjcf

        return mirror_mjcf


# 延迟导入
convert_urdf_to_mjcf_wrapper = None
mirror_mjcf = None


def rotate_and_elevate_mjcf(
    mjcf_path, output_path=None, rotation_angle=-90, elevation_height=1.1
):
    """
    旋转MJCF模型并抬高base_link

    Args:
        mjcf_path: 输入MJCF文件路径
        output_path: 输出MJCF文件路径，如果为None则覆盖输入文件
        rotation_angle: 旋转角度（度），正值为逆时针，负值为顺时针
        elevation_height: 抬高高度（米）

    Returns:
        bool: 操作是否成功
    """
    try:
        mjcf_path = Path(mjcf_path)
        if output_path is None:
            output_path = mjcf_path
        else:
            output_path = Path(output_path)

        # 解析XML
        tree = ET.parse(mjcf_path)
        root = tree.getroot()

        # 找到worldbody
        worldbody = root.find("worldbody")
        if worldbody is None:
            print("错误: 找不到worldbody元素")
            return False

        # 找到base_link
        base_link = None
        for body in worldbody.findall("body"):
            if body.attrib.get("name", "") == "base_link":
                base_link = body
                break

        if base_link is None:
            print("错误: 找不到base_link")
            return False

        # 将角度转换为弧度
        angle_rad = math.radians(rotation_angle)

        # 计算旋转四元数（绕X轴旋转）
        # 四元数格式: [w, x, y, z]
        w = math.cos(angle_rad / 2)
        x = math.sin(angle_rad / 2)
        y = 0
        z = 0

        # 创建旋转四元数字符串
        rotation_quat = f"{w:.8f} {x:.8f} {y:.8f} {z:.8f}"

        # 获取base_link的当前位置
        current_pos = base_link.attrib.get("pos", "0 0 0")
        pos_parts = current_pos.split()
        if len(pos_parts) >= 3:
            try:
                x_pos = float(pos_parts[0])
                y_pos = float(pos_parts[1])
                z_pos = float(pos_parts[2])

                # 抬高Z坐标
                z_pos += elevation_height

                # 更新位置
                new_pos = f"{x_pos:.8f} {y_pos:.8f} {z_pos:.8f}"
                base_link.attrib["pos"] = new_pos

                # 添加或更新旋转四元数
                base_link.attrib["quat"] = rotation_quat

                print(f"✓ 已将模型绕X轴旋转 {rotation_angle} 度并抬高到 {z_pos:.3f} 米")
                print(f"  旋转四元数: {rotation_quat}")
                print(f"  新位置: {new_pos}")

            except ValueError:
                print("错误: 无法解析base_link的位置坐标")
                return False
        else:
            # 如果base_link没有位置属性，添加新的位置和旋转
            new_pos = f"0.0 0.0 {elevation_height:.8f}"
            base_link.attrib["pos"] = new_pos
            base_link.attrib["quat"] = rotation_quat

            print(
                f"✓ 已添加绕X轴旋转 {rotation_angle} 度并设置高度为 {elevation_height:.3f} 米"
            )
            print(f"  旋转四元数: {rotation_quat}")
            print(f"  新位置: {new_pos}")

        # 保存文件
        tree.write(output_path, encoding="utf-8", xml_declaration=True)
        print(f"✓ 已保存修改后的MJCF文件: {output_path}")

        return True

    except Exception as e:
        print(f"✗ 旋转和抬高操作失败: {e}")
        import traceback

        traceback.print_exc()
        return False


def urdf2mjcf_full_pipeline(
    urdf_path, mjcf_right_path, mjcf_full_path, rotation_angle=-90, elevation_height=1.1
):
    """
    URDF转MJCF完整流程，包含旋转和抬高

    Args:
        urdf_path: URDF文件路径
        mjcf_right_path: 右腿MJCF输出路径
        mjcf_full_path: 完整双腿MJCF输出路径
        rotation_angle: 旋转角度（度），默认-90度
        elevation_height: 抬高高度（米），默认1.1米

    Returns:
        bool: 操作是否成功
    """
    # 延迟导入模块
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


def main():
    import os

    parser = argparse.ArgumentParser(
        description="URDF转双腿MJCF自动化工具，支持旋转和抬高模型",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
1. 完整转换流程（传统方式）:
   python urdf2mjcf.py --urdf input.urdf --mjcf output.xml --rotate -90 --elevate 1.1

2. 完整转换流程（新方式）:
   python urdf2mjcf.py convert --urdf input.urdf --mjcf output.xml --rotate -90 --elevate 1.1

3. 仅旋转和抬高现有MJCF文件:
   python urdf2mjcf.py rotate --input input.xml --output rotated.xml --rotate -90 --elevate 1.1

4. 旋转并覆盖原文件:
   python urdf2mjcf.py rotate --input input.xml --rotate -90 --elevate 1.1
""",
    )

    # 添加传统参数（保持向后兼容）
    parser.add_argument("--urdf", type=str, help="输入URDF文件路径（传统方式）")
    parser.add_argument("--mjcf", type=str, help="输出双腿MJCF文件路径（传统方式）")
    parser.add_argument(
        "--tmp", type=str, default=None, help="中间右腿MJCF文件路径（可选，传统方式）"
    )
    parser.add_argument(
        "--rotate",
        type=float,
        default=-90.0,
        help="绕X轴旋转角度（度），正值为逆时针，负值为顺时针，默认-90度",
    )
    parser.add_argument(
        "--elevate", type=float, default=1.1, help="抬高高度（米），默认1.1米"
    )

    # 创建子命令组
    subparsers = parser.add_subparsers(dest="command", help="选择操作模式")

    # 完整转换命令
    convert_parser = subparsers.add_parser("convert", help="完整URDF转MJCF流程")
    convert_parser.add_argument(
        "--urdf", type=str, required=True, help="输入URDF文件路径"
    )
    convert_parser.add_argument(
        "--mjcf", type=str, required=True, help="输出双腿MJCF文件路径"
    )
    convert_parser.add_argument(
        "--tmp", type=str, default=None, help="中间右腿MJCF文件路径（可选）"
    )
    convert_parser.add_argument(
        "--rotate",
        type=float,
        default=-90.0,
        help="绕X轴旋转角度（度），正值为逆时针，负值为顺时针，默认-90度",
    )
    convert_parser.add_argument(
        "--elevate", type=float, default=1.1, help="抬高高度（米），默认1.1米"
    )

    # 仅旋转命令
    rotate_parser = subparsers.add_parser("rotate", help="仅旋转和抬高现有MJCF文件")
    rotate_parser.add_argument(
        "--input", type=str, required=True, help="输入MJCF文件路径"
    )
    rotate_parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="输出MJCF文件路径（可选，默认覆盖输入）",
    )
    rotate_parser.add_argument(
        "--rotate",
        type=float,
        default=-90.0,
        help="绕X轴旋转角度（度），正值为逆时针，负值为顺时针，默认-90度",
    )
    rotate_parser.add_argument(
        "--elevate", type=float, default=1.1, help="抬高高度（米），默认1.1米"
    )

    args = parser.parse_args()

    # 判断使用哪种模式
    if args.command is None:
        # 传统模式：使用--urdf和--mjcf参数
        if args.urdf is None or args.mjcf is None:
            parser.print_help()
            print("\n错误: 传统模式需要 --urdf 和 --mjcf 参数")
            sys.exit(1)

        # 完整转换流程（传统方式）
        urdf_path = Path(args.urdf).resolve()
        mjcf_full_path = Path(args.mjcf).resolve()
        user_tmp = args.tmp is not None
        if user_tmp:
            mjcf_right_path = Path(args.tmp).resolve()
        else:
            mjcf_right_path = mjcf_full_path.parent / (
                mjcf_full_path.stem + "_right.xml"
            )

        print("使用传统模式进行完整转换...")
        urdf2mjcf_full_pipeline(
            urdf_path,
            mjcf_right_path,
            mjcf_full_path,
            rotation_angle=args.rotate,
            elevation_height=args.elevate,
        )

        # 自动删除中间右腿MJCF（仅当未指定--tmp时）
        if not user_tmp and mjcf_right_path.exists():
            try:
                os.remove(mjcf_right_path)
                print(f"已自动删除中间文件: {mjcf_right_path}")
            except Exception as e:
                print(f"自动删除中间文件失败: {e}")

    elif args.command == "convert":
        # 完整转换流程（新方式）
        urdf_path = Path(args.urdf).resolve()
        mjcf_full_path = Path(args.mjcf).resolve()
        user_tmp = args.tmp is not None
        if user_tmp:
            mjcf_right_path = Path(args.tmp).resolve()
        else:
            mjcf_right_path = mjcf_full_path.parent / (
                mjcf_full_path.stem + "_right.xml"
            )

        urdf2mjcf_full_pipeline(
            urdf_path,
            mjcf_right_path,
            mjcf_full_path,
            rotation_angle=args.rotate,
            elevation_height=args.elevate,
        )

        # 自动删除中间右腿MJCF（仅当未指定--tmp时）
        if not user_tmp and mjcf_right_path.exists():
            try:
                os.remove(mjcf_right_path)
                print(f"已自动删除中间文件: {mjcf_right_path}")
            except Exception as e:
                print(f"自动删除中间文件失败: {e}")

    elif args.command == "rotate":
        # 仅旋转和抬高现有MJCF文件
        input_path = Path(args.input).resolve()
        if not input_path.exists():
            print(f"错误: 输入文件不存在: {input_path}")
            sys.exit(1)

        output_path = args.output
        if output_path is None:
            output_path = input_path
        else:
            output_path = Path(output_path).resolve()

        print(f"旋转和抬高MJCF文件: {input_path}")
        print(f"绕X轴旋转角度: {args.rotate}度")
        print(f"抬高高度: {args.elevate}米")

        success = rotate_and_elevate_mjcf(
            input_path,
            output_path,
            rotation_angle=args.rotate,
            elevation_height=args.elevate,
        )

        if not success:
            print("✗ 旋转和抬高操作失败")
            sys.exit(1)

        print(f"✓ 操作完成，输出文件: {output_path}")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        # 如果没有参数，显示帮助信息
        print("URDF转MJCF工具，支持旋转和抬高模型")
        print("=" * 50)
        print("\n使用示例:")
        print("1. 完整转换流程:")
        print(
            "   python urdf2mjcf.py convert --urdf input.urdf --mjcf output.xml --rotate -90 --elevate 1.1"
        )
        print("\n2. 仅旋转和抬高现有MJCF文件:")
        print(
            "   python urdf2mjcf.py rotate --input input.xml --output rotated.xml --rotate -90 --elevate 1.1"
        )
        print("\n3. 旋转并覆盖原文件:")
        print(
            "   python urdf2mjcf.py rotate --input input.xml --rotate -90 --elevate 1.1"
        )
        print("\n4. 查看详细帮助:")
        print("   python urdf2mjcf.py convert --help")
        print("   python urdf2mjcf.py rotate --help")
        sys.exit(0)
    main()
