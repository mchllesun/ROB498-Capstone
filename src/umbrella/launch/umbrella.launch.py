from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    package_name = 'umbrella'

    return LaunchDescription([
        # # Odom republisher node
        # Node(
        #     package=package_name,
        #     executable='odom_republisher.py',
        #     name='odom_republisher',
        #     output='screen',
        # ),
        # # Comm node
        # Node(
        #     package=package_name,
        #     executable='comms_node.py',
        #     name='comms_node',
        #     output='screen',
        # ),
        # # Mission planner
        # Node(
        #     package=package_name,
        #     executable='mission_planner.py',
        #     name='mission_planner',
        #     output='screen',
        # ),
        # Camera driver
        Node(
            package=package_name,
            executable='imx_camera_driver.py',
            name='imx_camera_driver',
            output='screen',
        ),
        Node(
            package='image_transport',
            executable='republish',
            name='imx_republisher',
            arguments = ['raw', 'compressed'],
            remappings=[
                ('in', '/imx/image/raw'),
                ('out', '/imx/image/compressed'),
            ],
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