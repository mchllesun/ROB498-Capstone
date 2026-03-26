#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, PoseArray
from std_srvs.srv import Trigger
from enum import Enum
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

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

    HOVER_HEIGHT = 0.4       # meters

    def __init__(self):
        super().__init__('mission_planner')

        # --- FSM state ---
        self.state = State.IDLE
        self.target_pose = self._make_pose(z=-0.1)

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

        # --- State update timer at 2 Hz ---
        self.state_timer = self.create_timer(0.5, self._state_update)  # 2 Hz

        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # --- Subscribe to current position (mavros local position) ---
        self.pose_sub = self.create_subscription(
            PoseStamped, '/mavros/local_position/pose',
            self._cb_current_pose, qos_profile
        )

        # --- Waypoint subscriber ---
        self.waypoint_sub = self.create_subscription(
            PoseArray, 'mission_planner/waypoints', self._cb_waypoints, 10
        )

        # --- Dot tracker waypoint subscriber ---
        self.tracker_wp_sub = self.create_subscription(
            PoseStamped, '/waypoint_generator/target',
            self._cb_tracker_waypoint, 10
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

    # =====================================================================
    #  Helpers
    # =====================================================================
    def _make_pose(self, x=0.0, y=0.0, z=0.0):
        pose = PoseStamped()
        pose.header.frame_id = 'map'
        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.position.z = z
        # pose.pose.orientation.y = 1.0
        pose.pose.orientation.w = 1.0
        return pose

    def _transition(self, new_state: State) -> bool:
        """Attempt a state transition. Returns True if valid."""
        if new_state in TRANSITIONS.get(self.state, set()):
            old = self.state
            self.state = new_state
            self.get_logger().info(f'Transition: {old.value} -> {new_state.value}')
            self._on_enter(new_state)
            return True
        self.get_logger().warn(
            f'Invalid transition: {self.state.value} -> {new_state.value}'
        )
        return False

    # =====================================================================
    #  State entry actions — all side effects for entering a state live here
    # =====================================================================
    def _on_enter(self, state: State):
        if state == State.LAUNCHED:
            tpose = self._make_pose(z=self.HOVER_HEIGHT)
            # tpose.orientation.z = -1
            # tpose.orientation.w = 0
            self.target_pose = tpose

        elif state == State.WAYPOINT:
            wp = self.waypoints[self.waypoint_index]
            self.target_pose = self._make_pose(
                wp.pose.position.x,
                wp.pose.position.y,
                wp.pose.position.z,
            )
            self.get_logger().info(
                f'Waypoint {self.waypoint_index + 1}/{len(self.waypoints)}: '
                f'({wp.pose.position.x:.2f}, {wp.pose.position.y:.2f}, '
                f'{wp.pose.position.z:.2f})'
            )

        elif state == State.LANDING:
            self.target_pose = self._make_pose(z=-0.10)

        elif state == State.ABORTED:
            self.target_pose = self._make_pose(z=-0.2)

    # =====================================================================
    #  State update — periodic logic for each state (runs at 2 Hz)
    # =====================================================================
    def _state_update(self):
        pass

    # =====================================================================
    #  Subscriber callbacks
    # =====================================================================
    def _cb_current_pose(self, msg: PoseStamped):
        self.current_pose = msg

    def _publish_setpoint(self):
        self.target_pose.header.stamp = self.get_clock().now().to_msg()
        self.pose_pub.publish(self.target_pose)

    def _cb_waypoints(self, msg: PoseArray):
        if not len(self.waypoints) == 0:
            return
        for pose in msg.poses:
            ps = PoseStamped()
            ps.header.frame_id = 'map'
            ps.pose = pose
            ps.pose.position.x = -pose.position.x
            ps.pose.position.y = -pose.position.y

            # ps.pose.orientation.x = 0
            # ps.pose.orientation.y = 0
            # ps.pose.orientation.z = -1
            # ps.pose.orientation.w = 0
            self.waypoints.append(ps)
        self.waypoint_index = 0
        self.get_logger().info(f'Received {len(self.waypoints)} waypoints')

    def _cb_tracker_waypoint(self, msg: PoseStamped):
        """Follow the dot tracker target whenever the FSM is in WAYPOINT mode."""
        if self.state == State.WAYPOINT:
            self.target_pose = msg

    # =====================================================================
    #  Service callbacks — just validate and transition, no side effects
    # =====================================================================
    def _cb_launch(self, request, response):
        response.success = self._transition(State.LAUNCHED)
        return response

    def _cb_test(self, request, response):
        if not self.waypoints:
            self.get_logger().warn('No waypoints received yet')
            response.success = False
            response.message = 'No waypoints received'
            return response
        self.waypoint_index = 0
        response.success = self._transition(State.WAYPOINT)
        return response

    def _cb_land(self, request, response):
        response.success = self._transition(State.LANDING)
        return response

    def _cb_abort(self, request, response):
        response.success = self._transition(State.ABORTED)
        return response


def main(args=None):
    rclpy.init(args=args)
    node = MissionPlanner()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
