from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    package_name = 'flight_exercise_2'

    return LaunchDescription([
        Node(
            package=package_name,
            executable='offboard_test.py',
            name='offboard_test',
            output='screen',
        ),
    ])
