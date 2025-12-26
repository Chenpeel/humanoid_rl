#!/usr/bin/env python3
"""
MJCF 到 USD 转换脚本

将 MJCF (MuJoCo XML) 文件转换为 USD 格式，供 Isaac Lab 使用。

用法:
    python convert.py <mjcf_file> <output_usd> [options]

示例:
    python convert.py assets/xmls/models/jiyuan/jiyuan.xml assets/usd/jiyuan.usd --headless
    python convert.py assets/xmls/models/jiyuan/jiyuan.xml assets/usd/jiyuan.usd --headless --make-instanceable

参数说明:
    mjcf_file           输入的 MJCF 文件路径
    output_usd          输出的 USD 文件路径
    --headless          无头模式运行（推荐在服务器上使用）
    --make-instanceable 生成可实例化的 USD（推荐，用于多环境克隆）
    --fix-base          固定基座（默认 False，双足机器人需要自由移动）
    --import-sites      导入 MJCF 中的 sites（默认 True）
"""

import argparse
import os
import sys
from pathlib import Path

# 必须先启动 Isaac Sim 应用（在导入 Isaac Lab 之前）
from isaaclab.app import AppLauncher

# 创建参数解析器
parser = argparse.ArgumentParser(
    description="将 MJCF 文件转换为 USD 格式",
    formatter_class=argparse.RawDescriptionHelpFormatter,
)
parser.add_argument("input", type=str, help="输入的 MJCF 文件路径")
parser.add_argument("output", type=str, help="输出的 USD 文件路径")
parser.add_argument("--fix-base", action="store_true", default=False, help="固定基座（默认 False）")
parser.add_argument("--import-sites", action="store_true", default=True, help="导入 sites（默认 True）")
parser.add_argument(
    "--make-instanceable",
    action="store_true",
    default=True,
    help="生成可实例化的 USD（默认 True，推荐）",
)

# 添加 AppLauncher 参数
AppLauncher.add_app_launcher_args(parser)

# 解析参数
args = parser.parse_args()

# 启动 Isaac Sim 应用
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

"""导入 Isaac Lab 模块"""

import omni.kit.app

from isaaclab.sim.converters import MjcfConverter, MjcfConverterCfg
from isaaclab.utils.assets import check_file_path
from isaaclab.utils.dict import print_dict


def main():
    """主函数：执行 MJCF 到 USD 转换"""

    # 1. 验证输入文件路径
    mjcf_path = args.input
    if not os.path.isabs(mjcf_path):
        mjcf_path = os.path.abspath(mjcf_path)

    if not check_file_path(mjcf_path):
        print(f"错误: 输入文件不存在: {mjcf_path}")
        sys.exit(1)

    # 2. 准备输出文件路径
    output_path = args.output
    if not os.path.isabs(output_path):
        output_path = os.path.abspath(output_path)

    # 创建输出目录
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)

    # 3. 创建转换器配置
    mjcf_converter_cfg = MjcfConverterCfg(
        asset_path=mjcf_path,
        usd_dir=output_dir,
        usd_file_name=os.path.basename(output_path),
        fix_base=args.fix_base,  # False 用于双足机器人（需要自由移动）
        import_sites=args.import_sites,  # True 导入 MJCF 中的 sites
        make_instanceable=args.make_instanceable,  # True 用于多环境克隆
        force_usd_conversion=True,  # 强制重新转换
        self_collision=False,  # 禁用自碰撞（由 MJCF 中的碰撞组控制）
    )

    # 4. 打印转换信息
    print("=" * 80)
    print("MJCF 到 USD 转换")
    print("=" * 80)
    print(f"输入 MJCF 文件: {mjcf_path}")
    print(f"输出 USD 文件: {output_path}")
    print("\n转换器配置:")
    print_dict(mjcf_converter_cfg.to_dict(), nesting=0)
    print("=" * 80)

    # 5. 执行转换
    try:
        print("\n开始转换...")
        mjcf_converter = MjcfConverter(mjcf_converter_cfg)

        print("\n✓ 转换成功！")
        print(f"生成的 USD 文件: {mjcf_converter.usd_path}")

        # 检查文件大小
        usd_size = os.path.getsize(mjcf_converter.usd_path)
        print(f"文件大小: {usd_size / (1024 * 1024):.2f} MB")

        print("=" * 80)

    except Exception as e:
        print("\n✗ 转换失败！")
        print(f"错误信息: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    finally:
        # 关闭 Isaac Sim 应用
        simulation_app.close()
