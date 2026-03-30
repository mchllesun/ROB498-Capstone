from launch import LaunchDescription
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='umbrella',
            executable='obj_tracker_node.py',
            name='obj_tracker_node',
            output='screen',
        ),
        Node(
            package='umbrella',
            executable='imx_camera_driver.py',
            name='imx_camera_driver',
            output='screen',
        ),
    ])
