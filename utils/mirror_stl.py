#!/usr/bin/env python3
"""
镜像 STL 文件 - 用于生成左腿网格

将右腿 STL 文件在 X 轴上镜像，生成对应的左腿 STL 文件。
对应 MJCF 中的 scale="-1.0 1.0 1.0"

用法:
    python scripts/mirror_stl.py
"""

import struct
from pathlib import Path


def read_stl(filepath):
    """读取 STL 文件（二进制格式）"""
    with open(filepath, 'rb') as f:
        # 读取 80 字节头部
        header = f.read(80)

        # 读取三角形数量
        count = struct.unpack('I', f.read(4))[0]

        # 读取所有三角形
        triangles = []
        for _ in range(count):
            # 法向量 (3 floats)
            normal = struct.unpack('3f', f.read(12))
            # 顶点 1 (3 floats)
            v1 = struct.unpack('3f', f.read(12))
            # 顶点 2 (3 floats)
            v2 = struct.unpack('3f', f.read(12))
            # 顶点 3 (3 floats)
            v3 = struct.unpack('3f', f.read(12))
            # 属性字节计数 (2 bytes)
            attr = f.read(2)

            triangles.append({
                'normal': normal,
                'v1': v1,
                'v2': v2,
                'v3': v3,
                'attr': attr
            })

    return header, triangles


def mirror_stl(triangles, axis='x'):
    """镜像三角形网格

    Args:
        triangles: 三角形列表
        axis: 镜像轴 ('x', 'y', 或 'z')

    Returns:
        镜像后的三角形列表
    """
    axis_idx = {'x': 0, 'y': 1, 'z': 2}[axis]

    mirrored = []
    for tri in triangles:
        # 镜像顶点坐标
        v1 = list(tri['v1'])
        v2 = list(tri['v2'])
        v3 = list(tri['v3'])

        v1[axis_idx] *= -1
        v2[axis_idx] *= -1
        v3[axis_idx] *= -1

        # 镜像法向量
        normal = list(tri['normal'])
        normal[axis_idx] *= -1

        # 注意：由于顶点顺序改变，需要翻转法向量方向
        # 但在 X 轴镜像时，我们改变了顶点顺序，所以法向量也要相应调整
        # 实际上，为了保持正确的面朝向，我们需要交换两个顶点
        v2, v3 = v3, v2

        mirrored.append({
            'normal': tuple(normal),
            'v1': tuple(v1),
            'v2': tuple(v2),
            'v3': tuple(v3),
            'attr': tri['attr']
        })

    return mirrored


def write_stl(filepath, header, triangles):
    """写入 STL 文件（二进制格式）"""
    with open(filepath, 'wb') as f:
        # 写入头部
        f.write(header)

        # 写入三角形数量
        f.write(struct.pack('I', len(triangles)))

        # 写入所有三角形
        for tri in triangles:
            # 法向量
            f.write(struct.pack('3f', *tri['normal']))
            # 顶点 1
            f.write(struct.pack('3f', *tri['v1']))
            # 顶点 2
            f.write(struct.pack('3f', *tri['v2']))
            # 顶点 3
            f.write(struct.pack('3f', *tri['v3']))
            # 属性字节
            f.write(tri['attr'])


def main():
    """主函数"""
    # 定义项目路径
    project_root = Path(__file__).parent.parent
    meshes_dir = project_root / "assets" / "meshes"

    # 定义需要镜像的右腿 STL 文件
    right_stl_files = [
        "right_hip_pitch_engine_link.STL",
        "right_hip_yaw_engine_link.STL",
        "right_hip_roll_link.STL",
        "right_hip_cube_link.STL",
        "right_thigh_link.STL",
        "right_shin_link.STL",
        "right_ankle_1_3_link.STL",
        "right_ankle_1_3_cube_link.STL",
        "right_ankle_1_3_page_link.STL",
        "right_ankle_2_3_link.STL",
        "right_ankle_2_3_cube_link.STL",
        "right_ankle_2_3_page_link.STL",
        "right_ankle_3_3_link.STL",
        "right_ankle_3_3_cube_link.STL",
        "right_ankle_3_3_page_link.STL",
        "right_ankle_cube_link.STL",
        "right_ankle_axle_link.STL",
        "right_foot_link.STL",
        "right_toe_link.STL",
    ]

    print("开始镜像 STL 文件...")
    print(f"输入目录: {meshes_dir}")
    print()

    for filename in right_stl_files:
        right_path = meshes_dir / filename
        left_filename = filename.replace("right_", "left_")
        left_path = meshes_dir / left_filename

        if not right_path.exists():
            print(f"⚠️  跳过: {filename} (文件不存在)")
            continue

        print(f"处理: {filename} -> {left_filename}")

        # 读取右腿 STL
        header, triangles = read_stl(right_path)
        print(f"  读取: {len(triangles)} 个三角形")

        # 镜像 (在 X 轴上，对应 scale="-1.0 1.0 1.0")
        mirrored_triangles = mirror_stl(triangles, axis='x')

        # 写入左腿 STL
        write_stl(left_path, header, mirrored_triangles)
        print(f"  写入: {left_path}")

    print()
    print("✓ 完成！所有左腿 STL 文件已生成。")
    print()
    print("注意：此脚本对应 MJCF 中的 scale=\"-1.0 1.0 1.0\"")
    print("由于 Isaac Sim MJCF 导入器的 bug (https://github.com/isaac-sim/IsaacSim/issues/340)")
    print("我们必须预先镜像 STL 文件，而不是在 MJCF 中使用 scale 参数。")


if __name__ == "__main__":
    main()
