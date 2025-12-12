#!/usr/bin/env python3
"""
Utility Tools Manager - Unified interface for all utilities

Available tools:
- xml: MJCF XML modular split/merge
- urdf: URDF to MJCF conversion
- mirror: Mirror MJCF models (right leg -> left leg)

Usage:
    python -m utils <tool> <command> [options]

Examples:
    python -m utils xml split input.xml -o output_dir
    python -m utils xml merge input_dir -o output.xml
    python -m utils urdf convert input.urdf -o output.xml
    python -m utils mirror mjcf input.xml -o output.xml
"""

import sys
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description='Utility Tools Manager',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument('tool', choices=['xml', 'urdf', 'mirror'],
                        help='Tool to use')
    parser.add_argument('command', help='Command to execute')
    parser.add_argument('args', nargs='*', help='Command arguments')

    args, unknown = parser.parse_known_args()

    # Route to appropriate tool
    if args.tool == 'xml':
        from .xml_tools.xml_modular_tool import main as xml_main
        sys.argv = ['xml_modular_tool.py', args.command] + args.args + unknown
        xml_main()

    elif args.tool == 'urdf':
        from .urdf2mjcf.convert_tool import main as urdf_main
        sys.argv = ['convert_tool.py', args.command] + args.args + unknown
        urdf_main()

    elif args.tool == 'mirror':
        from .urdf2mjcf.mirror_tool import main as mirror_main
        sys.argv = ['mirror_tool.py', args.command] + args.args + unknown
        mirror_main()


if __name__ == '__main__':
    main()
