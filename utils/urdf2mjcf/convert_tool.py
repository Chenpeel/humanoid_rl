#!/usr/bin/env python3
"""
URDF to MJCF Conversion Tool

Converts URDF files to MJCF format with mesh path fixes.

Usage:
    python convert_tool.py convert <input.urdf> -o <output.xml>
"""

import sys
import re
import argparse
from pathlib import Path


def convert_urdf(input_urdf: Path, output_mjcf: Path):
    """Convert URDF to MJCF"""
    try:
        from urdf2mjcf.convert import run as urdf2mjcf_convert
    except ImportError:
        try:
            from urdf2mjcf import run as urdf2mjcf_convert
        except ImportError:
            print("Error: urdf2mjcf not installed")
            print("Install: pip install urdf2mjcf")
            sys.exit(1)

    print(f"[Convert] {input_urdf} → {output_mjcf}")

    # Convert
    urdf2mjcf_convert(str(input_urdf), str(output_mjcf))
    print("  ✓ Conversion complete")

    # Fix mesh paths
    fix_mesh_paths(output_mjcf)
    print("  ✓ Mesh paths fixed")

    print(f"\n✓ Complete: {output_mjcf}")


def fix_mesh_paths(mjcf_path: Path):
    """Fix mesh paths and remove empty material attributes"""
    content = mjcf_path.read_text(encoding='utf-8')

    # Replace package:// paths with relative paths
    content = re.sub(
        r'package://[^/]+/meshes/',
        '../meshes/',
        content
    )

    # Remove empty material attributes (MuJoCo doesn't allow empty material names)
    content = re.sub(
        r'\s+material=""',
        '',
        content
    )

    mjcf_path.write_text(content, encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(
        description='URDF to MJCF Conversion Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python convert_tool.py convert robot.urdf -o robot.xml
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Command')

    # Convert command
    convert = subparsers.add_parser('convert', help='Convert URDF to MJCF')
    convert.add_argument('input', type=str, help='Input URDF file')
    convert.add_argument('-o', '--output', type=str,
                         required=True, help='Output MJCF file')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    if args.command == 'convert':
        input_file = Path(args.input)
        output_file = Path(args.output)

        if not input_file.exists():
            print(f"Error: {input_file} not found")
            sys.exit(1)

        convert_urdf(input_file, output_file)


if __name__ == '__main__':
    main()
