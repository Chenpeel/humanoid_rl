#!/usr/bin/env python3
"""
MJCF Mirror Tool

Mirrors right leg to create left leg, generating bilateral model.

Usage:
    python mirror_tool.py mirror <input.xml> -o <output.xml>
"""

import copy
import sys
import argparse
import xml.etree.ElementTree as ET
from pathlib import Path


def mirror_pos(pos_str):
    """Mirror position across YZ plane"""
    vals = [float(x) for x in pos_str.strip().split()]
    if len(vals) == 3:
        vals[0] = -vals[0]
    return " ".join(f"{v:.8f}" for v in vals)


def mirror_axis(axis_str):
    """Mirror axis across YZ plane"""
    vals = [float(x) for x in axis_str.strip().split()]
    if len(vals) == 3:
        vals[0] = -vals[0]
    return " ".join(f"{v:.8f}" for v in vals)


def mirror_quat(quat_str):
    """Mirror quaternion across YZ plane"""
    vals = [float(x) for x in quat_str.strip().split()]
    if len(vals) == 4:
        vals[2] = -vals[2]
        vals[3] = -vals[3]
    return " ".join(f"{v:.8f}" for v in vals)


def mirror_name(name):
    """Replace right_ with left_"""
    return name.replace("right_", "left_")


def mirror_body(body, is_root=False):
    """Recursively mirror body and children"""
    body_copy = copy.deepcopy(body)

    if "name" in body_copy.attrib:
        body_copy.attrib["name"] = mirror_name(body_copy.attrib["name"])

    if "pos" in body_copy.attrib:
        body_copy.attrib["pos"] = mirror_pos(body_copy.attrib["pos"])

    if "quat" in body_copy.attrib:
        body_copy.attrib["quat"] = mirror_quat(body_copy.attrib["quat"])

    # Mirror all child elements
    for child in body_copy:
        if child.tag == "joint":
            if "name" in child.attrib:
                child.attrib["name"] = mirror_name(child.attrib["name"])
            if "axis" in child.attrib:
                child.attrib["axis"] = mirror_axis(child.attrib["axis"])

        elif child.tag == "geom":
            if "name" in child.attrib:
                child.attrib["name"] = mirror_name(child.attrib["name"])
            if "pos" in child.attrib:
                child.attrib["pos"] = mirror_pos(child.attrib["pos"])
            if "quat" in child.attrib:
                child.attrib["quat"] = mirror_quat(child.attrib["quat"])
            if "mesh" in child.attrib:
                child.attrib["mesh"] = mirror_name(child.attrib["mesh"])

        elif child.tag == "site":
            if "name" in child.attrib:
                child.attrib["name"] = mirror_name(child.attrib["name"])
            if "pos" in child.attrib:
                child.attrib["pos"] = mirror_pos(child.attrib["pos"])
            if "quat" in child.attrib:
                child.attrib["quat"] = mirror_quat(child.attrib["quat"])

        elif child.tag == "inertial":
            if "pos" in child.attrib:
                child.attrib["pos"] = mirror_pos(child.attrib["pos"])
            if "quat" in child.attrib:
                child.attrib["quat"] = mirror_quat(child.attrib["quat"])

        elif child.tag == "body":
            pass  # Recursive mirror handled later

    # Recursively mirror child bodies
    for child in body_copy.findall("body"):
        idx = list(body_copy).index(child)
        body_copy.remove(child)
        body_copy.insert(idx, mirror_body(child, is_root=False))

    return body_copy


def mirror_mjcf(input_file: Path, output_file: Path):
    """Mirror MJCF model"""
    print(f"[Mirror] {input_file} → {output_file}")

    tree = ET.parse(input_file)
    root = tree.getroot()

    # Mirror meshes in asset
    asset = root.find("asset")
    if asset is not None:
        right_meshes = []
        for mesh in asset.findall("mesh"):
            if "right_" in mesh.attrib.get("name", ""):
                right_meshes.append(mesh)

        for mesh in right_meshes:
            left_mesh = copy.deepcopy(mesh)
            left_mesh.attrib["name"] = mirror_name(mesh.attrib["name"])

            # Update file path
            if "file" in left_mesh.attrib:
                file_path = left_mesh.attrib["file"]
                left_mesh.attrib["file"] = file_path.replace("right_", "left_")

            # Add scale attribute for mirroring
            left_mesh.attrib["scale"] = "-1.0 1.0 1.0"

            asset.append(left_mesh)

        print("  ✓ Meshes mirrored")

    # Mirror robot bodies
    worldbody = root.find("worldbody")
    if worldbody is not None:
        base_link = worldbody.find(".//body[@name='base_link']")
        if base_link is not None:
            right_bodies = []
            for body in base_link.findall("body"):
                if "right_" in body.attrib.get("name", ""):
                    right_bodies.append(body)

            for body in right_bodies:
                left_body = mirror_body(body, is_root=True)
                base_link.append(left_body)

            print("  ✓ Bodies mirrored")

    # Mirror actuators
    actuator = root.find("actuator")
    if actuator is not None:
        right_actuators = []
        for motor in actuator.findall("motor"):
            if "right_" in motor.attrib.get("name", ""):
                right_actuators.append(motor)

        for motor in right_actuators:
            left_motor = copy.deepcopy(motor)
            left_motor.attrib["name"] = mirror_name(motor.attrib["name"])
            left_motor.attrib["joint"] = mirror_name(motor.attrib["joint"])
            actuator.append(left_motor)

        print("  ✓ Actuators mirrored")

    # Mirror contact exclusions
    contact = root.find("contact")
    if contact is not None:
        right_contacts = []
        for exclude in contact.findall("exclude"):
            body1 = exclude.attrib.get("body1", "")
            body2 = exclude.attrib.get("body2", "")
            if "right_" in body1 or "right_" in body2:
                right_contacts.append(exclude)

        for exclude in right_contacts:
            left_exclude = copy.deepcopy(exclude)
            left_exclude.attrib["body1"] = mirror_name(exclude.attrib["body1"])
            left_exclude.attrib["body2"] = mirror_name(exclude.attrib["body2"])
            contact.append(left_exclude)

        print("  ✓ Contact exclusions mirrored")

    # Write output
    tree.write(output_file, encoding="utf-8", xml_declaration=True)
    print(f"\n✓ Complete: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description='MJCF Mirror Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python mirror_tool.py mirror right_leg.xml -o bilateral.xml
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Command')

    # Mirror command
    mirror = subparsers.add_parser(
        'mirror', help='Mirror right leg to create left')
    mirror.add_argument('input', type=str, help='Input MJCF file')
    mirror.add_argument('-o', '--output', type=str,
                        required=True, help='Output MJCF file')

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

        mirror_mjcf(input_file, output_file)


if __name__ == '__main__':
    main()
