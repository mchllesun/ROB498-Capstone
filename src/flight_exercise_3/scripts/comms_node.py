#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseArray
from std_srvs.srv import Trigger

class CommNode(Node):
    def __init__(self):
        super().__init__('rob498_drone_11')

        # Service clients to mission_planner
        self.cli_launch = self.create_client(Trigger, 'mission_planner/launch')
        self.cli_test = self.create_client(Trigger, 'mission_planner/test')
        self.cli_land = self.create_client(Trigger, 'mission_planner/land')
        self.cli_abort = self.create_client(Trigger, 'mission_planner/abort')

        self.srv_launch = self.create_service(Trigger, 'rob498_drone_11/comm/launch', self.callback_launch)
        self.srv_test = self.create_service(Trigger, 'rob498_drone_11/comm/test', self.callback_test)
        self.srv_land = self.create_service(Trigger, 'rob498_drone_11/comm/land', self.callback_land)
        self.srv_abort = self.create_service(Trigger, 'rob498_drone_11/comm/abort', self.callback_abort)

        # Waypoint relay
        self.waypoint_pub = self.create_publisher(PoseArray, 'mission_planner/waypoints', 10)
        self.waypoint_sub = self.create_subscription(
            PoseArray, 'rob498_drone_11/comm/waypoints', self.callback_waypoints, 10
        )

        self.get_logger().info('Comm node started.')

    def callback_launch(self, request, response):
        self.get_logger().info('Launch requested. Forwarding to mission planner.')
        self.cli_launch.call_async(Trigger.Request())
        return response

    def callback_test(self, request, response):
        self.get_logger().info('Test requested. Forwarding to mission planner.')
        self.cli_test.call_async(Trigger.Request())
        return response

    def callback_land(self, request, response):
        self.get_logger().info('Land requested. Forwarding to mission planner.')
        self.cli_land.call_async(Trigger.Request())
        return response

    def callback_abort(self, request, response):
        self.get_logger().info('Abort requested. Forwarding to mission planner.')
        self.cli_abort.call_async(Trigger.Request())
        return response

    def callback_waypoints(self, msg: PoseArray):
        self.get_logger().info(f'Received {len(msg.poses)} waypoints. Forwarding to mission planner.')
        self.waypoint_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = CommNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()