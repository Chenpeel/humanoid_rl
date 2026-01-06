"""
MJCF镜像工具：将右腿MJCF通过YZ镜像自动生成左腿结构并插入，生成双腿MJCF
"""

import copy
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


# ============================================================================================
# ======================================= 镜像工具函数 =========================================
# ============================================================================================

def mirror_pos(pos_str):
    """镜像位置坐标 (x -> -x)"""
    vals = [float(x) for x in pos_str.strip().split()]
    if len(vals) == 3:
        vals[0] = -vals[0]
    return " ".join(f"{v:.8f}" for v in vals)

def mirror_axis(axis_str):
    """镜像轴向量 (x -> -x)"""
    vals = [float(x) for x in axis_str.strip().split()]
    if len(vals) == 3:
        vals[0] = -vals[0]
    return " ".join(f"{v:.8f}" for v in vals)

def mirror_quat(quat_str):
    """镜像四元数 (y -> -y, z -> -z)"""
    vals = [float(x) for x in quat_str.strip().split()]
    if len(vals) == 4:
        vals[2] = -vals[2]
        vals[3] = -vals[3]
    return " ".join(f"{v:.8f}" for v in vals)

def mirror_name(name):
    """镜像名称 (right_ -> left_)"""
    return name.replace("right_", "left_")

# ============================================================================================
# ===================================== END: 镜像工具函数 ======================================
# ============================================================================================


# ============================================================================================
# ======================================= 元素镜像逻辑 =========================================
# ============================================================================================

def mirror_body(body, is_root=False):
    """递归镜像body及其子元素"""
    body_copy = copy.deepcopy(body)
    if "name" in body_copy.attrib:
        body_copy.attrib["name"] = mirror_name(body_copy.attrib["name"])

    if "pos" in body_copy.attrib:
        body_copy.attrib["pos"] = mirror_pos(body_copy.attrib["pos"])
    if "quat" in body_copy.attrib:
        body_copy.attrib["quat"] = mirror_quat(body_copy.attrib["quat"])

    child_bodies = []
    for elem in list(body_copy):
        if elem.tag == "body":
            child_bodies.append(elem)

    for child_body in child_bodies:
        new_elem = mirror_body(child_body, is_root=False)
        for i, elem in enumerate(body_copy):
            if elem == child_body:
                body_copy[i] = new_elem
                break

    for elem in body_copy:
        if elem.tag == "joint":
            if "name" in elem.attrib:
                elem.attrib["name"] = mirror_name(elem.attrib["name"])
            if "axis" in elem.attrib:
                elem.attrib["axis"] = mirror_axis(elem.attrib["axis"])
            if "pos" in elem.attrib:
                elem.attrib["pos"] = mirror_pos(elem.attrib["pos"])
        elif elem.tag == "geom":
            if "name" in elem.attrib:
                elem.attrib["name"] = mirror_name(elem.attrib["name"])
            if "mesh" in elem.attrib:
                elem.attrib["mesh"] = mirror_name(elem.attrib["mesh"])
            if "pos" in elem.attrib:
                elem.attrib["pos"] = mirror_pos(elem.attrib["pos"])
            if "quat" in elem.attrib:
                elem.attrib["quat"] = mirror_quat(elem.attrib["quat"])
        elif elem.tag == "inertial":
            if "pos" in elem.attrib:
                elem.attrib["pos"] = mirror_pos(elem.attrib["pos"])
            if "quat" in elem.attrib:
                elem.attrib["quat"] = mirror_quat(elem.attrib["quat"])
        elif elem.tag == "site":
            if "name" in elem.attrib:
                elem.attrib["name"] = mirror_name(elem.attrib["name"])
            if "pos" in elem.attrib:
                elem.attrib["pos"] = mirror_pos(elem.attrib["pos"])
            if "quat" in elem.attrib:
                elem.attrib["quat"] = mirror_quat(elem.attrib["quat"])
        elif elem.tag == "camera":
            if "name" in elem.attrib:
                elem.attrib["name"] = mirror_name(elem.attrib["name"])
            if "pos" in elem.attrib:
                elem.attrib["pos"] = mirror_pos(elem.attrib["pos"])
            if "quat" in elem.attrib:
                elem.attrib["quat"] = mirror_quat(elem.attrib["quat"])
    return body_copy

# --------------------------------------------------------------------------------------------

def mirror_actuator(actuator):
    """镜像执行器"""
    actuator_copy = copy.deepcopy(actuator)
    for elem in actuator_copy:
        if "name" in elem.attrib:
            elem.attrib["name"] = mirror_name(elem.attrib["name"])
        if "joint" in elem.attrib:
            elem.attrib["joint"] = mirror_name(elem.attrib["joint"])
    return actuator_copy

# --------------------------------------------------------------------------------------------

def mirror_contact(contact):
    """镜像接触排除"""
    contact_copy = copy.deepcopy(contact)
    for elem in contact_copy:
        if "body1" in elem.attrib:
            elem.attrib["body1"] = mirror_name(elem.attrib["body1"])
        if "body2" in elem.attrib:
            elem.attrib["body2"] = mirror_name(elem.attrib["body2"])
    return contact_copy

# --------------------------------------------------------------------------------------------

def mirror_sensor(sensor):
    """镜像传感器"""
    sensor_copy = copy.deepcopy(sensor)
    for elem in sensor_copy:
        for k in list(elem.attrib.keys()):
            if "name" in k or "site" in k or "objname" in k:
                elem.attrib[k] = mirror_name(elem.attrib[k])
    return sensor_copy

# ============================================================================================
# ===================================== END: 元素镜像逻辑 ======================================
# ============================================================================================


# ============================================================================================
# ======================================= 主处理逻辑 ==========================================
# ============================================================================================

def mirror_mjcf(mjcf_path, output_path=None, left_leg_config=None):
    """
    读取右腿MJCF，镜像生成左腿，输出双腿MJCF
    """
    mjcf_path = Path(mjcf_path)
    if output_path is None:
        output_path = mjcf_path.parent / (mjcf_path.stem + "_mirrored.xml")
    else:
        output_path = Path(output_path)

    tree = ET.parse(mjcf_path)
    root = tree.getroot()

    worldbody = root.find("worldbody")
    base_link = None
    for body in worldbody.findall("body"):
        if body.attrib.get("name", "") == "base_link":
            base_link = body
            break
    if base_link is None:
        print("未找到base_link")
        sys.exit(1)

    right_leg_bodies = [
        child
        for child in base_link.findall("body")
        if child.attrib.get("name", "").startswith("right_")
    ]

    existing_left_body_names = set()
    for child in base_link.findall("body"):
        body_name = child.attrib.get("name", "")
        if body_name.startswith("left_"):
            existing_left_body_names.add(body_name)

    left_leg_bodies = [mirror_body(b, is_root=True) for b in right_leg_bodies]

    for left_body in left_leg_bodies:
        body_name = left_body.attrib.get("name", "")
        if body_name and body_name not in existing_left_body_names:
            base_link.append(left_body)
            existing_left_body_names.add(body_name)

    all_body_names = set()
    all_joint_names = set()
    all_site_names = set()

    def collect_element_names(element):
        if element.tag == "body" and "name" in element.attrib:
            all_body_names.add(element.attrib["name"])
        elif element.tag == "joint" and "name" in element.attrib:
            all_joint_names.add(element.attrib["name"])
        elif element.tag == "site" and "name" in element.attrib:
            all_site_names.add(element.attrib["name"])
        for child in element:
            collect_element_names(child)

    collect_element_names(worldbody)

    # 处理 Actuators
    actuator = root.find("actuator")
    left_actuator = mirror_actuator(actuator)
    existing_actuator_names = set()
    for elem in actuator:
        if "name" in elem.attrib:
            existing_actuator_names.add(elem.attrib["name"])

    for elem in left_actuator:
        elem_name = elem.attrib.get("name", "")
        joint_name = elem.attrib.get("joint", "")
        if (
            elem_name
            and elem_name not in existing_actuator_names
            and joint_name
            and joint_name in all_joint_names
        ):
            actuator.append(elem)
            existing_actuator_names.add(elem_name)

    # 处理 Contact
    contact = root.find("contact")
    left_contact = mirror_contact(contact)
    existing_contacts = set()
    for elem in contact:
        body1 = elem.attrib.get("body1", "")
        body2 = elem.attrib.get("body2", "")
        if body1 and body2:
            existing_contacts.add((body1, body2))

    for elem in left_contact:
        body1 = elem.attrib.get("body1", "")
        body2 = elem.attrib.get("body2", "")
        if (
            body1
            and body2
            and (body1, body2) not in existing_contacts
            and body1 in all_body_names
            and body2 in all_body_names
        ):
            contact.append(elem)
            existing_contacts.add((body1, body2))

    # 处理 Sensor
    sensor = root.find("sensor")
    if sensor is not None:
        left_sensor = mirror_sensor(sensor)
        existing_sensor_names = set()
        for elem in sensor:
            if "name" in elem.attrib:
                existing_sensor_names.add(elem.attrib["name"])

        for elem in left_sensor:
            elem_name = elem.attrib.get("name", "")
            site_name = elem.attrib.get("site", "")
            objname = elem.attrib.get("objname", "")

            has_valid_reference = False
            if site_name and site_name in all_site_names:
                has_valid_reference = True
            elif objname and (objname in all_site_names or objname in all_body_names):
                has_valid_reference = True

            if (
                elem_name
                and elem_name not in existing_sensor_names
                and has_valid_reference
            ):
                sensor.append(elem)
                existing_sensor_names.add(elem_name)

    # 处理 Assets
    asset = root.find("asset")
    existing_mesh_names = set()
    for m in asset.findall("mesh"):
        if "name" in m.attrib:
            existing_mesh_names.add(m.attrib["name"])

    mesh_names = [
        m.attrib["name"]
        for m in asset.findall("mesh")
        if m.attrib["name"].startswith("right_")
    ]
    for mesh_name in mesh_names:
        mesh = asset.find(f"mesh[@name='{mesh_name}']")
        mesh_copy = copy.deepcopy(mesh)
        left_mesh_name = mirror_name(mesh_copy.attrib["name"])

        if left_mesh_name not in existing_mesh_names:
            mesh_copy.attrib["name"] = left_mesh_name
            scale = [1.0, 1.0, 1.0]
            if "scale" in mesh_copy.attrib:
                scale = [float(x) for x in mesh_copy.attrib["scale"].split()]
            scale[0] = -scale[0]
            mesh_copy.attrib["scale"] = f"{scale[0]} {scale[1]} {scale[2]}"
            asset.append(mesh_copy)
            existing_mesh_names.add(left_mesh_name)

    tree.write(output_path, encoding="utf-8", xml_declaration=True)
    print(f"✓ 已生成双腿MJCF: {output_path}")

# ============================================================================================
# ===================================== END: 主处理逻辑 ========================================
# ============================================================================================