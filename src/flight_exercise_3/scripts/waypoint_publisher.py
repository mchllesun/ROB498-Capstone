#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseArray, Pose


class WaypointPublisher(Node):
    """Publishes a set of dummy waypoints to test the waypoint pipeline."""

    def __init__(self):
        super().__init__('waypoint_publisher')

        self.pub = self.create_publisher(
            PoseArray, 'rob498_drone_11/comm/waypoints', 10
        )

        msg = PoseArray()
        msg.header.frame_id = 'map'
        msg.header.stamp = self.get_clock().now().to_msg()

        # --- Define dummy waypoints (x, y, z) ---
        waypoint_coords = [
            (0.0, 0.0, 0.0),
            (-2.0, 0.0, 1.0),
            (1.0, -2.0, 1.5),
            (0.0, 1.0, 0.45),
        ]

        for x, y, z in waypoint_coords:
            pose = Pose()
            pose.position.x = x
            pose.position.y = y
            pose.position.z = z
            pose.orientation.w = 1.0
            msg.poses.append(pose)

        self.pub.publish(msg)
        self.get_logger().info(f'Published {len(msg.poses)} waypoints')


def main(args=None):
    rclpy.init(args=args)
    node = WaypointPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
