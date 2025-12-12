"""
Utility Tools Package

Provides command-line tools for robot model conversion and manipulation.

Modules:
    xml_tools: MJCF XML modular split/merge
    urdf2mjcf: URDF to MJCF conversion and mirroring
    urdf2ros2: URDF mirroring and ROS1→ROS2 migration
"""

__version__ = "1.0.0"

__all__ = [
    'xml_tools',
    'urdf2mjcf',
    'urdf2ros2',
]
