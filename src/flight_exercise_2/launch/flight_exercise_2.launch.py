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
        # Node(
        #     package='tf2_ros',
        #     executable='static_transform_publisher',
        #     name='t265_tf_pub',
        #     arguments = ['0', '0', '0', 
        #                  '-1.5707', '0', '0', 
        #                  'odom', 'odom_frame']
        # ),
        # Node(
        #     package='tf2_ros',
        #     executable='static_transform_publisher',
        #     name='map_to_odom_pub',
        #     arguments = ['0', '0', '0', 
        #                  '0', '0', '0', 
        #                  'map', 'odom']
        # ),
    ])