import xml.etree.ElementTree as ET
import copy
import math
import sys
import argparse
import os


def parse_xyz(xyz_str):
    return [float(x) for x in xyz_str.split()]


def parse_rpy(rpy_str):
    return [float(x) for x in rpy_str.split()]


def format_float(f):
    return f"{f:.6g}"


def format_xyz(xyz):
    return f"{format_float(xyz[0])} {format_float(xyz[1])} {format_float(xyz[2])}"


def format_rpy(rpy):
    return f"{format_float(rpy[0])} {format_float(rpy[1])} {format_float(rpy[2])}"

# Euler (RPY) to Quaternion (w, x, y, z)


def rpy_to_quat(r, p, y):
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

# Quaternion to Euler (RPY)


def quat_to_rpy(w, x, y, z):
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
    # Convert to quat
    q = rpy_to_quat(rpy[0], rpy[1], rpy[2])
    # Mirror quat: w, x, -y, -z (Mirror across YZ plane / X-axis reflection)
    q_mirror = [q[0], q[1], -q[2], -q[3]]
    # Convert back to rpy
    return quat_to_rpy(q_mirror[0], q_mirror[1], q_mirror[2], q_mirror[3])


def mirror_urdf(input_file, output_file):
    tree = ET.parse(input_file)
    robot = tree.getroot()

    # Rename robot
    if 'name' in robot.attrib:
        robot.attrib['name'] = robot.attrib['name'] + "_full"

    new_elements = []

    # Find all links and joints that start with "right_"
    # We will duplicate them and convert to "left_"

    # Collect existing names to avoid duplicates if run multiple times (though we create new file)
    existing_names = set()
    for elem in robot:
        if 'name' in elem.attrib:
            existing_names.add(elem.attrib['name'])

    for elem in list(robot):  # Iterate over copy of list
        name = elem.attrib.get('name', '')
        if name.startswith('right_'):
            new_elem = copy.deepcopy(elem)
            new_name = name.replace('right_', 'left_')
            new_elem.attrib['name'] = new_name

            if new_name in existing_names:
                continue

            # Process Link
            if new_elem.tag == 'link':
                # Mirror Inertial
                inertial = new_elem.find('inertial')
                if inertial is not None:
                    origin = inertial.find('origin')
                    if origin is not None and 'xyz' in origin.attrib:
                        xyz = parse_xyz(origin.attrib['xyz'])
                        xyz[0] = -xyz[0]  # Flip X
                        origin.attrib['xyz'] = format_xyz(xyz)
                        # RPY of inertial origin? Usually 0 0 0, but if not, should mirror it too.
                        if 'rpy' in origin.attrib:
                            rpy = parse_rpy(origin.attrib['rpy'])
                            rpy = mirror_rpy(rpy)
                            origin.attrib['rpy'] = format_rpy(rpy)

                    inertia = inertial.find('inertia')
                    if inertia is not None:
                        # Ixy -> -Ixy, Ixz -> -Ixz
                        if 'ixy' in inertia.attrib:
                            inertia.attrib['ixy'] = format_float(
                                -float(inertia.attrib['ixy']))
                        if 'ixz' in inertia.attrib:
                            inertia.attrib['ixz'] = format_float(
                                -float(inertia.attrib['ixz']))

                # Mirror Visuals
                for visual in new_elem.findall('visual'):
                    # Origin
                    origin = visual.find('origin')
                    if origin is not None:
                        if 'xyz' in origin.attrib:
                            xyz = parse_xyz(origin.attrib['xyz'])
                            xyz[0] = -xyz[0]
                            origin.attrib['xyz'] = format_xyz(xyz)
                        if 'rpy' in origin.attrib:
                            rpy = parse_rpy(origin.attrib['rpy'])
                            rpy = mirror_rpy(rpy)
                            origin.attrib['rpy'] = format_rpy(rpy)

                    # Geometry Mesh
                    geometry = visual.find('geometry')
                    if geometry is not None:
                        mesh = geometry.find('mesh')
                        if mesh is not None:
                            # Keep filename same (pointing to right mesh)
                            # Add/Update scale
                            scale = [1.0, 1.0, 1.0]
                            if 'scale' in mesh.attrib:
                                scale = [float(x)
                                         for x in mesh.attrib['scale'].split()]

                            # Mirror X scale
                            scale[0] = -scale[0]
                            mesh.attrib['scale'] = f"{scale[0]} {scale[1]} {scale[2]}"

                # Mirror Collisions
                for collision in new_elem.findall('collision'):
                    # Origin
                    origin = collision.find('origin')
                    if origin is not None:
                        if 'xyz' in origin.attrib:
                            xyz = parse_xyz(origin.attrib['xyz'])
                            xyz[0] = -xyz[0]
                            origin.attrib['xyz'] = format_xyz(xyz)
                        if 'rpy' in origin.attrib:
                            rpy = parse_rpy(origin.attrib['rpy'])
                            rpy = mirror_rpy(rpy)
                            origin.attrib['rpy'] = format_rpy(rpy)

                    # Geometry Mesh
                    geometry = collision.find('geometry')
                    if geometry is not None:
                        mesh = geometry.find('mesh')
                        if mesh is not None:
                            scale = [1.0, 1.0, 1.0]
                            if 'scale' in mesh.attrib:
                                scale = [float(x)
                                         for x in mesh.attrib['scale'].split()]
                            scale[0] = -scale[0]
                            mesh.attrib['scale'] = f"{scale[0]} {scale[1]} {scale[2]}"

            # Process Joint
            elif new_elem.tag == 'joint':
                # Parent
                parent = new_elem.find('parent')
                if parent is not None and parent.attrib['link'].startswith('right_'):
                    parent.attrib['link'] = parent.attrib['link'].replace(
                        'right_', 'left_')

                # Child
                child = new_elem.find('child')
                if child is not None and child.attrib['link'].startswith('right_'):
                    child.attrib['link'] = child.attrib['link'].replace(
                        'right_', 'left_')

                # Origin
                origin = new_elem.find('origin')
                if origin is not None:
                    if 'xyz' in origin.attrib:
                        xyz = parse_xyz(origin.attrib['xyz'])
                        xyz[0] = -xyz[0]  # Flip X
                        origin.attrib['xyz'] = format_xyz(xyz)

                    if 'rpy' in origin.attrib:
                        rpy = parse_rpy(origin.attrib['rpy'])
                        rpy = mirror_rpy(rpy)
                        origin.attrib['rpy'] = format_rpy(rpy)

                # Axis
                # Axis direction usually kept same if we want symmetric behavior in world frame?
                # But let's check if we need to mirror axis.
                # If we mirror the frame (via RPY mirror), the local axis vector might need adjustment?
                # If axis is defined in local frame.
                # Our RPY mirror (w, x, -y, -z) mirrors the frame basis vectors:
                # X' = X
                # Y' = -Y
                # Z' = -Z
                # (This is a 180 deg rotation around X).
                # So if the original axis was (ax, ay, az) in local frame.
                # In the new frame (which is rotated 180 X relative to the "geometric mirror" frame?),
                # this is getting confusing.
                # Let's stick to: We mirrored the frame orientation using the quaternion flip.
                # This quaternion flip corresponds to mirroring the rotation across YZ plane.
                # So if we have an axis vector v in local frame.
                # We probably want to mirror the axis vector components too?
                # v_new = (vx, -vy, -vz)?
                # Let's check assets.csv again.
                # Right Hip Roll Axis: -1 0 0. Left: -1 0 0. (vx same).
                # Right Hip Pitch Axis: 0 -1 0. Left: 0 -1 0. (vy same).
                # This implies axis components are NOT flipped?
                # Wait. If frame is rotated 180 X.
                # Then local Y points in opposite direction of original local Y (relative to the mirrored geometry).
                # If we want the axis to point in same physical direction (e.g. World Y).
                # And local Y is flipped. Then we need to flip axis Y component?
                # If axis is (0, -1, 0) (points along -Y).
                # New axis (0, -1, 0) points along -Y'.
                # Since Y' = -Y_world (roughly). Then -Y' = Y_world.
                # So (0, -1, 0) in new frame points along +Y_world.
                # But original (0, -1, 0) points along -Y_world.
                # So the axis direction FLIPPED in world frame.
                # Is this what we want?
                # For Pitch: Right leg pitch usually rotates around Y.
                # If we want both legs to move forward with +q.
                # Right leg: +q moves leg forward.
                # Left leg: +q moves leg forward.
                # Since they are mirrored, "forward" is same direction.
                # So axis in world frame should be same.
                # If my frame transformation flips the world-direction of the local axis, I need to compensate.
                # Let's assume the user wants standard URDF mirroring.
                # I will apply the axis mirror: vx -> vx, vy -> -vy, vz -> -vz.
                # Let's try this.
                axis = new_elem.find('axis')
                if axis is not None and 'xyz' in axis.attrib:
                    xyz = parse_xyz(axis.attrib['xyz'])
                    # Mirror axis components: x -> x, y -> -y, z -> -z
                    # Because our frame mirror (quat w,x,-y,-z) flips Y and Z axes of the frame.
                    xyz[1] = -xyz[1]
                    xyz[2] = -xyz[2]
                    axis.attrib['xyz'] = format_xyz(xyz)

            new_elements.append(new_elem)

    # Append new elements
    for elem in new_elements:
        robot.append(elem)

    tree.write(output_file, encoding='utf-8', xml_declaration=True)
    print(f"Generated mirrored URDF: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Mirror URDF from right to left.")
    parser.add_argument("--input", required=True,
                        help="Input URDF file (with right leg)")
    parser.add_argument("--output", required=True, help="Output URDF file")
    args = parser.parse_args()

    mirror_urdf(args.input, args.output)
