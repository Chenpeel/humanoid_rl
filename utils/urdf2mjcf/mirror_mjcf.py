"""
MJCF镜像工具：将右腿MJCF通过YZ镜像自动生成左腿结构并插入，生成双腿MJCF
"""

import copy
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


def mirror_pos(pos_str):
    vals = [float(x) for x in pos_str.strip().split()]
    if len(vals) == 3:
        vals[0] = -vals[0]
    return " ".join(f"{v:.8f}" for v in vals)


def mirror_axis(axis_str):
    vals = [float(x) for x in axis_str.strip().split()]
    if len(vals) == 3:
        vals[0] = -vals[0]
    return " ".join(f"{v:.8f}" for v in vals)


def mirror_quat(quat_str):
    vals = [float(x) for x in quat_str.strip().split()]
    if len(vals) == 4:
        vals[2] = -vals[2]
        vals[3] = -vals[3]
    return " ".join(f"{v:.8f}" for v in vals)


def mirror_name(name):
    return name.replace("right_", "left_")


def mirror_body(body, is_root=False):
    """递归镜像body及其子元素
    Args:
        body: 要镜像的body元素
        is_root: 是否是根部body（直接连在base_link下）
    """
    body_copy = copy.deepcopy(body)
    if "name" in body_copy.attrib:
        body_copy.attrib["name"] = mirror_name(body_copy.attrib["name"])

    # 镜像pos和quat
    if "pos" in body_copy.attrib:
        body_copy.attrib["pos"] = mirror_pos(body_copy.attrib["pos"])
    if "quat" in body_copy.attrib:
        body_copy.attrib["quat"] = mirror_quat(body_copy.attrib["quat"])

    # 先收集所有子body元素
    child_bodies = []
    for elem in list(body_copy):  # 使用list()创建副本进行遍历
        if elem.tag == "body":
            child_bodies.append(elem)

    # 处理所有子body
    for child_body in child_bodies:
        new_elem = mirror_body(child_body, is_root=False)
        # 找到并替换子body
        for i, elem in enumerate(body_copy):
            if elem == child_body:
                body_copy[i] = new_elem
                break

    # 处理其他元素
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


def mirror_actuator(actuator):
    actuator_copy = copy.deepcopy(actuator)
    for elem in actuator_copy:
        if "name" in elem.attrib:
            elem.attrib["name"] = mirror_name(elem.attrib["name"])
        if "joint" in elem.attrib:
            elem.attrib["joint"] = mirror_name(elem.attrib["joint"])
    return actuator_copy


def mirror_contact(contact):
    contact_copy = copy.deepcopy(contact)
    for elem in contact_copy:
        if "body1" in elem.attrib:
            elem.attrib["body1"] = mirror_name(elem.attrib["body1"])
        if "body2" in elem.attrib:
            elem.attrib["body2"] = mirror_name(elem.attrib["body2"])
    return contact_copy


def mirror_sensor(sensor):
    sensor_copy = copy.deepcopy(sensor)
    for elem in sensor_copy:
        # 镜像所有属性中包含'name', 'site', 'objname'的值
        for k in list(elem.attrib.keys()):
            if "name" in k or "site" in k or "objname" in k:
                elem.attrib[k] = mirror_name(elem.attrib[k])
    return sensor_copy


def mirror_mjcf(mjcf_path, output_path=None, left_leg_config=None):
    """
    读取右腿MJCF，镜像生成左腿，输出双腿MJCF
    :param mjcf_path: 输入右腿MJCF路径
    :param output_path: 输出双腿MJCF路径，若为None则覆盖输入
    :param left_leg_config: 左腿配置字典，包含左腿根部body的pos和quat信息
                           格式: {'left_hip_yaw_link': {'pos': '0.08 0 0', 'quat': '...'}, ...}
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

    # 收集已存在的左腿body名称，避免重复添加
    existing_left_body_names = set()
    for child in base_link.findall("body"):
        body_name = child.attrib.get("name", "")
        if body_name.startswith("left_"):
            existing_left_body_names.add(body_name)

    left_leg_bodies = [mirror_body(b, is_root=True) for b in right_leg_bodies]

    # 不使用虚拟骨盆，直接添加左右腿到base_link
    # 左腿采用X镜像位置（X反向，YZ保持不变）
    for left_body in left_leg_bodies:
        body_name = left_body.attrib.get("name", "")
        if body_name and body_name not in existing_left_body_names:
            base_link.append(left_body)
            existing_left_body_names.add(body_name)

    # 收集模型中所有body、joint和site的名称
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

    actuator = root.find("actuator")
    left_actuator = mirror_actuator(actuator)

    # 收集已存在的actuator名称，避免重复添加
    existing_actuator_names = set()
    for elem in actuator:
        if "name" in elem.attrib:
            existing_actuator_names.add(elem.attrib["name"])

    # 只添加不存在的左腿actuator，并且对应的joint必须存在
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

    contact = root.find("contact")
    left_contact = mirror_contact(contact)

    # 收集已存在的contact排除对，避免重复添加
    existing_contacts = set()
    for elem in contact:
        body1 = elem.attrib.get("body1", "")
        body2 = elem.attrib.get("body2", "")
        if body1 and body2:
            existing_contacts.add((body1, body2))

    # 只添加不存在的左腿contact排除对，并且两个body都必须存在
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

    sensor = root.find("sensor")
    if sensor is not None:
        left_sensor = mirror_sensor(sensor)

        # 收集已存在的sensor名称，避免重复添加
        existing_sensor_names = set()
        for elem in sensor:
            if "name" in elem.attrib:
                existing_sensor_names.add(elem.attrib["name"])

        # 只添加不存在的左腿sensor，并且对应的site或objname必须存在
        for elem in left_sensor:
            elem_name = elem.attrib.get("name", "")
            site_name = elem.attrib.get("site", "")
            objname = elem.attrib.get("objname", "")

            # 检查sensor是否有效：要么有site引用，要么有objname引用
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

    asset = root.find("asset")

    # 收集已存在的mesh名称，避免重复添加
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

        # 只添加不存在的左腿mesh
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
