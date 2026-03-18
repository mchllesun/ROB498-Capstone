from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='flight_exercise_3',
            executable='waypoint_publisher.py',
            name='waypoint_publisher',
            output='screen',
        ),
    ])
