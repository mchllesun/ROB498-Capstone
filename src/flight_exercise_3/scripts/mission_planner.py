#!/usr/bin/env python3

import math
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, PoseArray
from std_srvs.srv import Trigger
from enum import Enum

class State(Enum):
    IDLE = 'IDLE'
    LAUNCHED = 'LAUNCHED'
    WAYPOINT = 'WAYPOINT'
    LANDING = 'LANDING'
    ABORTED = 'ABORTED'

# State transition table
TRANSITIONS = {
    State.IDLE:     {State.LAUNCHED},
    State.LAUNCHED: {State.WAYPOINT, State.LANDING, State.ABORTED},
    State.WAYPOINT: {State.LANDING, State.ABORTED},
    State.LANDING:  {State.IDLE, State.LAUNCHED, State.ABORTED},
    State.ABORTED:  {State.IDLE, State.LAUNCHED},
}

class MissionPlanner(Node):

    HOVER_HEIGHT = 0.33       # meters
    WAYPOINT_THRESHOLD = 0.15  # meters — distance to consider waypoint reached

    def __init__(self):
        super().__init__('mission_planner')

        # --- FSM state ---
        self.state = State.IDLE
        self.target_pose = PoseStamped()

        # --- Waypoint tracking ---
        self.waypoints = []       # list of PoseStamped
        self.waypoint_index = 0

        # --- Current drone position ---
        self.current_pose = None  # PoseStamped

        # --- Publisher: setpoint position at 20 Hz ---
        self.pose_pub = self.create_publisher(
            PoseStamped, 'mavros/setpoint_position/local', 10
        )
        self.timer = self.create_timer(0.05, self._publish_setpoint)  # 20 Hz

        # --- Waypoint check timer at 2 Hz ---
        self.waypoint_timer = self.create_timer(0.5, self._check_waypoint)  # 2 Hz

        # --- Subscribe to current position (mavros local position) ---
        self.pose_sub = self.create_subscription(
            PoseStamped, 'mavros/vision_pose/pose',
            self._cb_current_pose, 10
        )

        # --- Waypoint subscriber ---
        self.waypoint_sub = self.create_subscription(
            PoseArray, 'mission_planner/waypoints', self._cb_waypoints, 10
        )

        # --- Services ---
        self.srv_launch = self.create_service(
            Trigger, 'mission_planner/launch', self._cb_launch
        )
        self.srv_test = self.create_service(
            Trigger, 'mission_planner/test', self._cb_test
        )
        self.srv_land = self.create_service(
            Trigger, 'mission_planner/land', self._cb_land
        )
        self.srv_abort = self.create_service(
            Trigger, 'mission_planner/abort', self._cb_abort
        )

        self.get_logger().info(f'Mission planner started. State: {self.state.value}')

    # ---------- helpers ----------
    def _make_pose(self, x=0.0, y=0.0, z=0.0):
        pose = PoseStamped()
        pose.header.frame_id = 'map'
        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.position.z = z
        pose.pose.orientation.w = 1.0
        return pose

    def _distance_to(self, pose: PoseStamped) -> float:
        """Euclidean distance from current position to a target pose."""
        if self.current_pose is None:
            return float('inf')
        dx = self.current_pose.pose.position.x - pose.pose.position.x
        dy = self.current_pose.pose.position.y - pose.pose.position.y
        dz = self.current_pose.pose.position.z - pose.pose.position.z
        return math.sqrt(dx * dx + dy * dy + dz * dz)


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

    # ---------- callbacks ----------
    def _cb_current_pose(self, msg: PoseStamped):
        self.current_pose = msg

    def _publish_setpoint(self):
        self.target_pose.header.stamp = self.get_clock().now().to_msg()
        self.pose_pub.publish(self.target_pose)

    def _check_waypoint(self):
        """Check if we've reached the current waypoint (runs at 2 Hz)."""
        if self.state == State.WAYPOINT and self.current_pose is not None:
            dist = self._distance_to(self.target_pose)
            if dist < self.WAYPOINT_THRESHOLD:
                self._advance_waypoint()

    def _cb_waypoints(self, msg: PoseArray):
        self.waypoints = []
        for pose in msg.poses:
            ps = PoseStamped()
            ps.header.frame_id = 'map'
            ps.pose = pose
            self.waypoints.append(ps)
        self.waypoint_index = 0
        self.get_logger().info(f'Received {len(self.waypoints)} waypoints')

    def _set_waypoint_target(self):
        """Set target_pose to current waypoint with orientation facing the next one."""
        wp = self.waypoints[self.waypoint_index]
        self.target_pose = self._make_pose(
            wp.pose.position.x,
            wp.pose.position.y,
            wp.pose.position.z,
        )
        self.get_logger().info(
            f'Waypoint {self.waypoint_index + 1}/{len(self.waypoints)}: '
            f'({wp.pose.position.x:.2f}, {wp.pose.position.y:.2f}, {wp.pose.position.z:.2f})'
        )

    def _advance_waypoint(self):
        """Advance to the next waypoint if available."""
        if self.waypoint_index + 1 < len(self.waypoints):
            self.waypoint_index += 1
            self._set_waypoint_target()
        else:
            self.get_logger().info('All waypoints visited — holding last position')

    # ---------- service callbacks ----------
    def _cb_launch(self, request, response):
        if self._transition(State.LAUNCHED):
            self.target_pose = self._make_pose(z=self.HOVER_HEIGHT)
            response.success = True
        else:
            response.success = False
        return response

    def _cb_test(self, request, response):
        if not self.waypoints:
            self.get_logger().warn('No waypoints received yet')
            response.success = False
            response.message = 'No waypoints received'
            return response
        if self._transition(State.WAYPOINT):
            self.waypoint_index = 0
            self._set_waypoint_target()
            response.success = True
        else:
            response.success = False
        return response

    def _cb_land(self, request, response):
        if self._transition(State.LANDING):
            self.target_pose = self._make_pose(z=-0.10)
            response.success = True
        else:
            response.success = False
        return response

    def _cb_abort(self, request, response):
        if self._transition(State.ABORTED):
            self.target_pose = self._make_pose(z=-0.2)
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
