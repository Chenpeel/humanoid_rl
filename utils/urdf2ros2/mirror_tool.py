#!/usr/bin/env python3
"""
URDF Mirror Tool

Mirrors URDF models across YZ plane for bilateral robots.

Usage:
    python mirror_tool.py mirror <input.urdf> -o <output.urdf>
"""

import xml.etree.ElementTree as ET
import copy
import math
import sys
import argparse
from pathlib import Path


def rpy_to_quat(r, p, y):
    """Convert Euler angles (RPY) to quaternion (w, x, y, z)"""
    cy = math.cos(y * 0.5)
    sy = math.sin(y * 0.5)
    cp = math.cos(p * 0.5)
    sp = math.sin(p * 0.5)
    cr = math.cos(r * 0.5)
    sr = math.sin(r * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return [w, x, y, z]


def quat_to_rpy(w, x, y, z):
    """Convert quaternion (w, x, y, z) to Euler angles (RPY)"""
    ysqr = y * y

    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + ysqr)
    roll = math.atan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch = math.asin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (ysqr + z * z)
    yaw = math.atan2(t3, t4)

    return [roll, pitch, yaw]


def mirror_rpy(rpy):
    """Mirror RPY across YZ plane"""
    q = rpy_to_quat(rpy[0], rpy[1], rpy[2])
    q_mirror = [q[0], q[1], -q[2], -q[3]]
    return quat_to_rpy(q_mirror[0], q_mirror[1], q_mirror[2], q_mirror[3])


def mirror_urdf(input_file: Path, output_file: Path):
    """Mirror URDF model"""
    print(f"[Mirror] {input_file} → {output_file}")

    tree = ET.parse(input_file)
    robot = tree.getroot()

    # Update robot name
    robot_name = robot.get('name', 'robot')
    if 'right' in robot_name.lower():
        robot.set('name', robot_name.replace('right', 'bilateral'))
    else:
        robot.set('name', f'{robot_name}_bilateral')

    print("  ✓ Robot name updated")

    # Collect right links and joints
    right_links = []
    right_joints = []

    for link in robot.findall('link'):
        link_name = link.get('name', '')
        if 'right_' in link_name:
            right_links.append(link)

    for joint in robot.findall('joint'):
        joint_name = joint.get('name', '')
        if 'right_' in joint_name:
            right_joints.append(joint)

    print(f"  ✓ Found {len(right_links)} right links")
    print(f"  ✓ Found {len(right_joints)} right joints")

    # Mirror links
    for link in right_links:
        left_link = copy.deepcopy(link)
        left_link.set('name', link.get('name', '').replace('right_', 'left_'))

        # Update visual/collision mesh references
        for visual in left_link.findall('.//visual/geometry/mesh'):
            filename = visual.get('filename', '')
            visual.set('filename', filename.replace('right_', 'left_'))

        for collision in left_link.findall('.//collision/geometry/mesh'):
            filename = collision.get('filename', '')
            collision.set('filename', filename.replace('right_', 'left_'))

        robot.append(left_link)

    print(f"  ✓ Mirrored {len(right_links)} links")

    # Mirror joints
    for joint in right_joints:
        left_joint = copy.deepcopy(joint)
        left_joint.set('name', joint.get('name', '').replace('right_', 'left_'))

        # Update parent/child
        parent = left_joint.find('parent')
        if parent is not None:
            parent.set('link', parent.get('link', '').replace('right_', 'left_'))

        child = left_joint.find('child')
        if child is not None:
            child.set('link', child.get('link', '').replace('right_', 'left_'))

        # Mirror origin
        origin = left_joint.find('origin')
        if origin is not None:
            xyz_str = origin.get('xyz', '0 0 0')
            xyz = [float(x) for x in xyz_str.split()]
            xyz[0] = -xyz[0]  # Mirror X coordinate
            origin.set('xyz', f"{xyz[0]:.6g} {xyz[1]:.6g} {xyz[2]:.6g}")

            rpy_str = origin.get('rpy', '0 0 0')
            rpy = [float(x) for x in rpy_str.split()]
            rpy_mirrored = mirror_rpy(rpy)
            origin.set('rpy', f"{rpy_mirrored[0]:.6g} {rpy_mirrored[1]:.6g} {rpy_mirrored[2]:.6g}")

        # Mirror axis
        axis = left_joint.find('axis')
        if axis is not None:
            xyz_str = axis.get('xyz', '1 0 0')
            xyz = [float(x) for x in xyz_str.split()]
            xyz[0] = -xyz[0]  # Mirror X component
            axis.set('xyz', f"{xyz[0]:.6g} {xyz[1]:.6g} {xyz[2]:.6g}")

        robot.append(left_joint)

    print(f"  ✓ Mirrored {len(right_joints)} joints")

    # Write output
    tree.write(output_file, encoding='utf-8', xml_declaration=True)
    print(f"\n✓ Complete: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description='URDF Mirror Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python mirror_tool.py mirror right_leg.urdf -o bilateral.urdf
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Command')

    # Mirror command
    mirror = subparsers.add_parser('mirror', help='Mirror URDF across YZ plane')
    mirror.add_argument('input', type=str, help='Input URDF file')
    mirror.add_argument('-o', '--output', type=str, required=True, help='Output URDF file')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    if args.command == 'mirror':
        input_file = Path(args.input)
        output_file = Path(args.output)

        if not input_file.exists():
            print(f"Error: {input_file} not found")
            sys.exit(1)

        mirror_urdf(input_file, output_file)


if __name__ == '__main__':
    main()
