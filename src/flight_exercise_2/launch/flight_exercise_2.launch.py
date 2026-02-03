from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    package_name = 'flight_exercise_2'

    return LaunchDescription([
        # Odom republisher node
        Node(
            package=package_name,
            executable='odom_republisher.py',
            name='odom_republisher',
            output='screen',
        ),
        # Comm node
        Node(
            package=package_name,
            executable='comms_node.py',
            name='comms_node',
            output='screen',
        ),
    ])