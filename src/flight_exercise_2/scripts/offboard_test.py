#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped


class OffboardTest(Node):
    def __init__(self):
        super().__init__('offboard_test')

        self.publisher_ = self.create_publisher(
            PoseStamped, 'mavros/setpoint_position/local', 10
        )
        self.timer = self.create_timer(0.05, self.timer_callback)  # 20 Hz

        self.get_logger().info('Offboard test node started.')

    def timer_callback(self):
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'

        # Position
        msg.pose.position.x = 0.0
        msg.pose.position.y = 0.0
        msg.pose.position.z = 0.5

        # Orientation (identity quaternion)
        msg.pose.orientation.x = 0.0
        msg.pose.orientation.y = 0.0
        msg.pose.orientation.z = 0.0
        msg.pose.orientation.w = 1.0

        self.publisher_.publish(msg)
        self.get_logger().info(
            f'Publishing pose: ({msg.pose.position.x}, {msg.pose.position.y}, {msg.pose.position.z})'
        )


def main(args=None):
    rclpy.init(args=args)
    node = OffboardTest()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
