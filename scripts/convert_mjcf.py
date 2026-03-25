#!/usr/bin/env python3
"""基于已安装的 Isaac Lab 包将 MJCF 转成 USD。"""

from __future__ import annotations

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert MJCF to USD with installed Isaac Lab packages.")
    parser.add_argument("input", type=Path, help="输入 MJCF/XML 文件路径")
    parser.add_argument("output", type=Path, help="输出 USD 文件路径")
    parser.add_argument("--make-instanceable", action="store_true", help="生成可实例化的 USD")
    parser.add_argument("--fix-base", action="store_true", help="固定根链接")
    parser.add_argument("--import-sites", action="store_true", help="导入 MJCF sites")
    parser.add_argument("--force", action="store_true", help="强制重新生成 USD")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    input_path = args.input.expanduser().resolve()
    output_path = args.output.expanduser().resolve()

    if not input_path.is_file():
        raise FileNotFoundError(f"输入文件不存在: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    app_launcher = AppLauncher(headless=True)
    simulation_app = app_launcher.app

    try:
        from isaaclab.sim.converters import MjcfConverter, MjcfConverterCfg

        cfg = MjcfConverterCfg(
            asset_path=str(input_path),
            usd_dir=str(output_path.parent),
            usd_file_name=output_path.name,
            force_usd_conversion=args.force,
            make_instanceable=args.make_instanceable,
            fix_base=args.fix_base,
            import_sites=args.import_sites,
        )
        converter = MjcfConverter(cfg)
        print(converter.usd_path)
        return 0
    finally:
        simulation_app.close()


if __name__ == "__main__":
    raise SystemExit(main())
