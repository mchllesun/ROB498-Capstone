#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from std_srvs.srv import Trigger
from enum import Enum


class State(Enum):
    IDLE = 'IDLE'
    LAUNCHED = 'LAUNCHED'
    LANDING = 'LANDING'
    ABORTED = 'ABORTED'


# State transition table
# Current State -> Allowed transitions
TRANSITIONS = {
    State.IDLE:     {State.LAUNCHED},
    State.LAUNCHED: {State.LANDING, State.ABORTED},
    State.LANDING:  {State.IDLE, State.LAUNCHED, State.ABORTED},
    State.ABORTED:  {State.IDLE, State.LAUNCHED},
}


class MissionPlanner(Node):

    HOVER_HEIGHT = 0.33  # meters

    def __init__(self):
        super().__init__('mission_planner')

        # --- FSM state ---
        self.state = State.IDLE
        self.target_pose = PoseStamped()

        # --- Publisher: setpoint position at 20 Hz ---
        self.pose_pub = self.create_publisher(
            PoseStamped, 'mavros/setpoint_position/local', 10
        )
        self.timer = self.create_timer(0.05, self._publish_setpoint)  # 20 Hz

        # --- Services ---
        self.srv_launch = self.create_service(
            Trigger, 'mission_planner/launch', self._cb_launch
        )
        self.srv_land = self.create_service(
            Trigger, 'mission_planner/land', self._cb_land
        )
        self.srv_abort = self.create_service(
            Trigger, 'mission_planner/abort', self._cb_abort
        )

        self.get_logger().info(f'Mission planner started. State: {self.state.value}')

    def _make_empty_pose(self):
        pose = PoseStamped()
        pose.header.frame_id = 'map'
        pose.pose.orientation.w = 1.0
        return pose

    def _transition(self, new_state: State) -> bool:
        """Attempt a state transition. Returns True if valid."""
        if new_state in TRANSITIONS.get(self.state, set()):
            old = self.state
            self.state = new_state
            self.get_logger().info(f'Transition: {old.value} -> {new_state.value}')
            return True
        self.get_logger().warn(
            f'Invalid transition: {self.state.value} -> {new_state.value}'
        )
        return False

    def _publish_setpoint(self):
        self.target_pose.header.stamp = self.get_clock().now().to_msg()
        self.pose_pub.publish(self.target_pose)

    def _cb_launch(self, request, response):
        if self._transition(State.LAUNCHED):
            self.target_pose = self._make_empty_pose()
            self.target_pose.pose.position.z = self.HOVER_HEIGHT
            response.success = True
        else:
            response.success = False
        return response

    def _cb_land(self, request, response):
        if self._transition(State.LANDING):
            self.target_pose = self._make_empty_pose()
            self.target_pose.pose.position.z = -0.02
            response.success = True
        else:
            response.success = False
        return response

    def _cb_abort(self, request, response):
        if self._transition(State.ABORTED):
            self.target_pose = self._make_empty_pose()
            self.target_pose.pose.position.z = -0.05
            response.success = True
        else:
            response.success = False
        return response


def main(args=None):
    rclpy.init(args=args)
    node = MissionPlanner()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
