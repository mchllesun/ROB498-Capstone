from launch import LaunchDescription
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    model_path = PathJoinSubstitution([
        FindPackageShare('umbrella'),
        'models',
        'yolo26n.pt',
    ])

    return LaunchDescription([
        # Camera driver
        Node(
            package='umbrella',
            executable='imx_camera_driver.py',
            name='imx_camera_driver',
            output='screen',
        ),
        # YOLO detector (Node 1: images -> detections)
        Node(
            package='umbrella',
            executable='yolo_detector_node.py',
            name='yolo_detector',
            output='screen',
            additional_env={'LD_PRELOAD': '/usr/lib/aarch64-linux-gnu/libgomp.so.1'},
            parameters=[{
                'image_topic': '/imx/image/raw',
                'detections_topic': '/yolo/detections',
                'model_path': model_path,
                'device': 'cuda:0',
                'use_fp16': False,
                'imgsz': 320,
                'conf_threshold': 0.35,
            }],
        ),
        # Person follower (Node 2: detections + pose -> setpoints)
        Node(
            package='umbrella',
            executable='person_follower_node.py',
            name='person_follower',
            output='screen',
            parameters=[{
                'detections_topic': '/yolo/detections',
                'pose_topic': '/mavros/local_position/pose',
                'setpoint_topic': '/mavros/setpoint_position/local',
                'target_class': 'person',
                'image_width': 640,
                'image_height': 640,
                'hover_height_m': 1.5,
                'kp_image_x': 0.6,
                'kp_image_y': 0.6,
                'max_xy_step_m': 0.15,
                'center_deadband': 0.08,
                'min_detection_confidence': 0.35,
            }],
        ),
    ])
