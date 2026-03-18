#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_srvs.srv import Empty, Trigger

class CommNode(Node):
    def __init__(self):
        super().__init__('rob498_drone_11')

        # Service clients to mission_planner
        self.cli_launch = self.create_client(Trigger, 'mission_planner/launch')
        self.cli_land = self.create_client(Trigger, 'mission_planner/land')
        self.cli_abort = self.create_client(Trigger, 'mission_planner/abort')

        self.srv_launch = self.create_service(Trigger, 'rob498_drone_11/comm/launch', self.callback_launch)
        self.srv_test = self.create_service(Trigger, 'rob498_drone_11/comm/test', self.callback_test)
        self.srv_land = self.create_service(Trigger, 'rob498_drone_11/comm/land', self.callback_land)
        self.srv_abort = self.create_service(Trigger, 'rob498_drone_11/comm/abort', self.callback_abort)

        self.get_logger().info(f'Comm node started.')

    def callback_launch(self, request, response):
        print('Launch Requested. Your drone should take off.')
        self.cli_launch.call_async(Trigger.Request())
        return response

    def callback_test(self, request, response):
        print('Test Requested. Your drone should perform the required tasks. Recording starts now.')
        return response

    def callback_land(self, request, response):
        print('Land Requested. Your drone should land.')
        self.cli_land.call_async(Trigger.Request())
        return response

    def callback_abort(self, request, response):
        print('Abort Requested. Your drone should land immediately due to safety considerations')
        self.cli_abort.call_async(Trigger.Request())
        return response

def main(args=None):
    rclpy.init(args=args)
    node = CommNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == "__main__":
    main()