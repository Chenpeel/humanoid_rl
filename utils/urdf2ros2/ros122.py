import os
import sys
import shutil
import xml.etree.ElementTree as ET


def convert_package_xml(package_path):
    xml_path = os.path.join(package_path, 'package.xml')
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Update buildtool_depend
    buildtool_found = False
    for buildtool in root.findall('buildtool_depend'):
        if buildtool.text == 'catkin':
            buildtool.text = 'ament_cmake'
            buildtool_found = True

    if not buildtool_found:
        # If no catkin found, maybe it's already ros2 or weird, but let's add ament_cmake
        bt = ET.SubElement(root, 'buildtool_depend')
        bt.text = 'ament_cmake'

    # Update depends
    # Remove old ROS1 depends
    for dep in list(root.findall('depend')):
        root.remove(dep)

    # Add ROS2 depends
    depends = [
        'rclpy',
        'urdf',
        'xacro',
        'robot_state_publisher',
        'joint_state_publisher_gui',
        'rviz2'
    ]
    for dep_name in depends:
        elem = ET.SubElement(root, 'depend')
        elem.text = dep_name

    # Update export
    export = root.find('export')
    if export is None:
        export = ET.SubElement(root, 'export')

    # Remove architecture_independent if exists
    for child in list(export):
        if child.tag == 'architecture_independent':
            export.remove(child)

    build_type = export.find('build_type')
    if build_type is None:
        build_type = ET.SubElement(export, 'build_type')
    build_type.text = 'ament_cmake'

    tree.write(xml_path, encoding='utf-8', xml_declaration=True)
    print(f"Converted {xml_path}")


def convert_cmakelists(package_path, package_name):
    cmake_path = os.path.join(package_path, 'CMakeLists.txt')
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

# Install directories
install(DIRECTORY config launch meshes urdf
  DESTINATION share/${{PROJECT_NAME}}
)

ament_package()
"""
    with open(cmake_path, 'w') as f:
        f.write(content)
    print(f"Converted {cmake_path}")


def create_launch_file(package_path, package_name):
    launch_dir = os.path.join(package_path, 'launch')
    if not os.path.exists(launch_dir):
        os.makedirs(launch_dir)

    # Find default URDF
    urdf_dir = os.path.join(package_path, 'urdf')
    default_urdf = f"{package_name}.urdf"
    if os.path.exists(urdf_dir):
        urdfs = [f for f in os.listdir(urdf_dir) if f.endswith('.urdf')]
        if urdfs:
            # Prefer the one matching package name, else first one
            if default_urdf in urdfs:
                pass
            else:
                default_urdf = urdfs[0]

    # Check for rviz config
    rviz_config_name = 'urdf.rviz'
    # We will move it to config/ later, so we assume it will be there

    launch_content = f"""import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

def generate_launch_description():
    pkg_share = get_package_share_directory('{package_name}')
    default_model_path = os.path.join(pkg_share, 'urdf', '{default_urdf}')
    default_rviz_config_path = os.path.join(pkg_share, 'config', '{rviz_config_name}')

    return LaunchDescription([
        DeclareLaunchArgument(
            'model', 
            default_value=default_model_path,
            description='Absolute path to robot urdf file'),
            
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            parameters=[{{'robot_description': ParameterValue(Command(['xacro ', LaunchConfiguration('model')]), value_type=str)}}]
        ),
        
        Node(
            package='joint_state_publisher_gui',
            executable='joint_state_publisher_gui',
            name='joint_state_publisher_gui'
        ),
        
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            output='screen',
            arguments=['-d', default_rviz_config_path]
        )
    ])
"""
    launch_path = os.path.join(launch_dir, 'display.launch.py')
    with open(launch_path, 'w') as f:
        f.write(launch_content)
    print(f"Created {launch_path}")

    # Move urdf.rviz to config if it exists in root
    rviz_config_src = os.path.join(package_path, 'urdf.rviz')
    config_dir = os.path.join(package_path, 'config')
    if not os.path.exists(config_dir):
        os.makedirs(config_dir)

    if os.path.exists(rviz_config_src):
        shutil.move(rviz_config_src, os.path.join(config_dir, 'urdf.rviz'))
        print(f"Moved urdf.rviz to config/urdf.rviz")
    elif not os.path.exists(os.path.join(config_dir, 'urdf.rviz')):
        # Create an empty one or warning?
        # Rviz2 can start without config, but we passed -d.
        # If file doesn't exist, rviz might complain or just start default.
        # Let's just warn.
        print("Warning: urdf.rviz not found in package root. Rviz might start with default config.")


def main():
    if len(sys.argv) < 2:
        print("Usage: python ros122.py <package_path>")
        sys.exit(1)

    package_path = sys.argv[1]
    package_path = os.path.abspath(package_path)

    if not os.path.exists(os.path.join(package_path, 'package.xml')):
        print(f"Error: No package.xml found in {package_path}")
        sys.exit(1)

    tree = ET.parse(os.path.join(package_path, 'package.xml'))
    root = tree.getroot()
    name_elem = root.find('name')
    if name_elem is None or name_elem.text is None:
        print("Error: Could not determine package name from package.xml")
        sys.exit(1)
    package_name = name_elem.text

    print(f"Converting package '{package_name}' to ROS2...")

    convert_package_xml(package_path)
    convert_cmakelists(package_path, package_name)
    create_launch_file(package_path, package_name)

    print("Done! You can now build with 'colcon build' and run 'ros2 launch <package> display.launch.py'")


if __name__ == '__main__':
    main()
