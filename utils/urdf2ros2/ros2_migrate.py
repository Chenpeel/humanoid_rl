#!/usr/bin/env python3
"""
ROS1 to ROS2 Migration Tool

Converts ROS1 packages to ROS2 format.

Usage:
    python ros2_migrate.py convert <package_path>
"""

import os
import sys
import argparse
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path


def convert_package_xml(package_path: Path):
    """Convert package.xml from ROS1 to ROS2 format"""
    xml_path = package_path / 'package.xml'

    if not xml_path.exists():
        print(f"  ✗ package.xml not found")
        return False

    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Update buildtool_depend
    buildtool_found = False
    for buildtool in root.findall('buildtool_depend'):
        if buildtool.text == 'catkin':
            buildtool.text = 'ament_cmake'
            buildtool_found = True

    if not buildtool_found:
        bt = ET.SubElement(root, 'buildtool_depend')
        bt.text = 'ament_cmake'

    # Remove old ROS1 dependencies
    for dep in list(root.findall('depend')):
        root.remove(dep)

    # Add ROS2 dependencies
    ros2_deps = [
        'rclpy',
        'urdf',
        'xacro',
        'robot_state_publisher',
        'joint_state_publisher_gui',
        'rviz2'
    ]

    for dep_name in ros2_deps:
        elem = ET.SubElement(root, 'depend')
        elem.text = dep_name

    # Update export
    export = root.find('export')
    if export is None:
        export = ET.SubElement(root, 'export')

    build_type = export.find('build_type')
    if build_type is None:
        build_type = ET.SubElement(export, 'build_type')
    build_type.text = 'ament_cmake'

    tree.write(xml_path, encoding='utf-8', xml_declaration=True)
    print(f"  ✓ package.xml converted")
    return True


def convert_cmakelists(package_path: Path, package_name: str):
    """Generate ROS2 CMakeLists.txt"""
    cmake_path = package_path / 'CMakeLists.txt'

    content = f"""cmake_minimum_required(VERSION 3.5)
project({package_name})

# Default to C99
if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)

# Install URDF files
install(DIRECTORY urdf
  DESTINATION share/${{PROJECT_NAME}}
)

# Install mesh files
install(DIRECTORY meshes
  DESTINATION share/${{PROJECT_NAME}}
)

# Install launch files
install(DIRECTORY launch
  DESTINATION share/${{PROJECT_NAME}}
)

ament_package()
"""

    cmake_path.write_text(content, encoding='utf-8')
    print(f"  ✓ CMakeLists.txt generated")
    return True


def convert_launch_files(package_path: Path):
    """Convert launch files from ROS1 to ROS2 format"""
    launch_dir = package_path / 'launch'

    if not launch_dir.exists():
        print(f"  ℹ No launch directory found")
        return True

    # Note: This is a simplified conversion
    # Full conversion would require more sophisticated parsing
    print(f"  ⚠ Launch files need manual conversion to Python format")
    print(f"    See: https://docs.ros.org/en/foxy/How-To-Guides/Migrating-from-ROS1/Migrating-Launch-Files.html")
    return True


def migrate_package(package_path: Path):
    """Migrate ROS1 package to ROS2"""
    print(f"[Migrate] {package_path}")

    if not package_path.exists():
        print(f"Error: Package path not found")
        return False

    package_xml = package_path / 'package.xml'
    if not package_xml.exists():
        print(f"Error: Not a ROS package (no package.xml)")
        return False

    # Get package name
    tree = ET.parse(package_xml)
    root = tree.getroot()
    package_name = root.find('name').text

    print(f"\n📦 Package: {package_name}")

    # Convert package.xml
    print("\n🔧 Converting package.xml...")
    if not convert_package_xml(package_path):
        return False

    # Convert CMakeLists.txt
    print("\n🔧 Generating CMakeLists.txt...")
    if not convert_cmakelists(package_path, package_name):
        return False

    # Check launch files
    print("\n🔧 Checking launch files...")
    convert_launch_files(package_path)

    print(f"\n✓ Migration complete!")
    print(f"\n📝 Next steps:")
    print(f"  1. Review and test the converted files")
    print(f"  2. Manually convert .launch files to Python format")
    print(f"  3. Update any package-specific dependencies")
    print(f"  4. Build: colcon build --packages-select {package_name}")

    return True


def main():
    parser = argparse.ArgumentParser(
        description='ROS1 to ROS2 Migration Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python ros2_migrate.py convert ~/catkin_ws/src/my_robot
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Command')

    # Convert command
    convert = subparsers.add_parser('convert', help='Convert ROS1 package to ROS2')
    convert.add_argument('package', type=str, help='Package directory path')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    if args.command == 'convert':
        package_path = Path(args.package)

        if not migrate_package(package_path):
            sys.exit(1)


if __name__ == '__main__':
    main()
