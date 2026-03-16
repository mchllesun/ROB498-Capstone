from launch import LaunchDescription
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='flight_exercise_3',
            executable='person_follow_yolo.py',
            name='person_follow_yolo',
            output='screen',
            parameters=[{
                'image_topic': '/imx/image/raw',
                'pose_topic': '/mavros/local_position/pose',
                'setpoint_topic': '/mavros/setpoint_position/local',
                'model_path': PathJoinSubstitution([
                    FindPackageShare('flight_exercise_3'),
                    'models',
                    'yolo26n.pt',
                ]),
                'use_fp16': True,
                'device': 'cuda:0',
                'imgsz': 320,
                'hover_height_m': 1.5,
                'kp_image_x': 0.6,
                'kp_image_y': 0.6,
                'max_xy_step_m': 0.15,
                'center_deadband': 0.08,
                'min_detection_confidence': 0.35,
            }],
        ),
    ])
