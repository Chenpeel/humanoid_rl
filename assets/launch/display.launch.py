import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

def generate_launch_description():
    pkg_share = get_package_share_directory('assets')
    default_model_path = os.path.join(pkg_share, 'urdf', 'assets.urdf')
    default_rviz_config_path = os.path.join(pkg_share, 'config', 'urdf.rviz')

    return LaunchDescription([
        DeclareLaunchArgument(
            'model', 
            default_value=default_model_path,
            description='Absolute path to robot urdf file'),
            
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            parameters=[{'robot_description': ParameterValue(Command(['xacro ', LaunchConfiguration('model')]), value_type=str)}]
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
