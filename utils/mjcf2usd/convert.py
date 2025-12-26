#!/usr/bin/env python3
"""
MJCF to USD 转换脚本

将 MJCF (MuJoCo XML) 文件转换为 USD 格式，供 Isaac Lab 使用。

用法:
    python convert.py <mjcf_file> [output_usd]

示例:
    python convert.py robot.xml robot.usd
    python convert.py assets/xmls/models/jiyuan/index.xml
"""

import argparse
from pathlib import Path
import sys

# 必须先启动 Isaac Sim 应用（在导入 Isaac Lab 之前）
from isaaclab.app import AppLauncher

# 创建参数解析器（用于 AppLauncher）
app_launcher_parser = argparse.ArgumentParser(add_help=False)
AppLauncher.add_app_launcher_args(app_launcher_parser)
app_launcher_args, remaining_args = app_launcher_parser.parse_known_args()

# 启动 Isaac Sim 应用（headless 模式）
app_launcher = AppLauncher(app_launcher_args)
simulation_app = app_launcher.app


def convert_mjcf_to_usd(mjcf_path: str, output_path: str = None) -> str:
    """将 MJCF 文件转换为 USD 格式

    Args:
        mjcf_path: MJCF 文件路径
        output_path: 输出 USD 文件路径（可选）

    Returns:
        生成的 USD 文件路径
    """
    from omni.isaac.sim.utils import create_usd_from_mjcf_file

    mjcf_file = Path(mjcf_path)
    if not mjcf_file.exists():
        raise FileNotFoundError(f"MJCF 文件不存在: {mjcf_path}")

    # 默认输出路径
    if output_path is None:
        output_path = str(mjcf_file.with_suffix(".usd"))
    else:
        output_path = str(Path(output_path))

    print(f"转换 MJCF: {mjcf_path}")
    print(f"输出 USD: {output_path}")

    # 使用 Isaac Sim 的 MJCF 导入功能
    usd_path = create_usd_from_mjcf_file(
        mjcf_file=str(mjcf_file.absolute()),
        usd_path=str(Path(output_path).absolute()),
        fix_base=False,  # 不固定基座
        import_sites=True,  # 导入 site
        self_collision=False,
    )

    print(f"✓ 转换完成: {usd_path}")
    return usd_path


def main():
    if len(remaining_args) < 1:
        print(__doc__)
        print("错误: 缺少 MJCF 文件参数")
        sys.exit(1)

    mjcf_file = remaining_args[0]
    output_file = remaining_args[1] if len(remaining_args) > 1 else None

    try:
        convert_mjcf_to_usd(mjcf_file, output_file)
    except FileNotFoundError as e:
        print(f"错误: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"转换失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        # 关闭 Isaac Sim 应用
        simulation_app.close()


if __name__ == "__main__":
    main()
