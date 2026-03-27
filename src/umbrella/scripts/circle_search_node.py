#!/usr/bin/env python3
"""
Circle Search Node

Monitors whether the dot tracker is seeing the red dot. If no detection is
received for TIMEOUT_SEC seconds while the drone is in tracking mode (signalled
by mission_planner publishing circle_search/enabled = True), this node:

  1. Captures the drone's current position as the circle centre.
  2. Publishes /circle_search/active = True so mission_planner enters CIRCLE_SEARCH.
  3. Streams waypoints along a 30 cm radius circle at the current altitude.
  4. After one full revolution it publishes active = False (mission_planner → hover).
  5. If a red-dot detection arrives at any time during the circle it immediately
     cancels the search by publishing active = False, allowing the normal tracker
     waypoints to take over again.

Topics
------
Subscribes:
  /umbrella/detections          (DetectionArray)  – raw detector output
  /mavros/local_position/pose   (PoseStamped)     – drone position
  /circle_search/enabled        (Bool)            – set True by mission_planner
                                                    when it enters WAYPOINT state
Publishes:
  /circle_search/target         (PoseStamped)     – circle waypoints
  /circle_search/active         (Bool)            – True while searching
"""

import math
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Bool
from umbrella.msg import DetectionArray

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
TIMEOUT_SEC    = 5.0   # seconds without detection before triggering search
RADIUS_M       = 0.30  # circle radius in metres
NUM_STEPS      = 36    # waypoints per revolution (every 10°)
STEP_ACCEPT_M  = 0.05  # metres within which a waypoint is considered reached
WP_RATE_HZ     = 4.0  # how often to check whether to advance the waypoint
# ---------------------------------------------------------------------------


class CircleSearchNode(Node):

    def __init__(self):
        super().__init__('circle_search_node')

        self._enabled        = False   # True when mission_planner is in WAYPOINT mode
        self._searching      = False   # True while actively flying the circle
        self._last_det_time  = None    # rclpy.time.Time of last non-empty detection
        self._current_pose   = None    # latest PoseStamped from mavros

        # Circle state
        self._circle_centre  = None   # (x, y, z) tuple captured at search start
        self._step_index     = 0      # current waypoint index within the circle
        self._circle_yaw_start = 0.0  # drone yaw when search triggered (unused but kept for reference)

        # QoS
        be_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        # --- Subscribers ---
        self._det_sub = self.create_subscription(
            DetectionArray, '/umbrella/detections', self._cb_detections, 10
        )
        self._pose_sub = self.create_subscription(
            PoseStamped, '/mavros/local_position/pose', self._cb_pose, be_qos
        )
        self._enabled_sub = self.create_subscription(
            Bool, '/circle_search/enabled', self._cb_enabled, 10
        )

        # --- Publishers ---
        self._target_pub = self.create_publisher(PoseStamped, '/circle_search/target', 10)
        self._active_pub = self.create_publisher(Bool, '/circle_search/active', 10)

        # --- Timers ---
        # Timeout check at 2 Hz
        self._timeout_timer = self.create_timer(0.5, self._check_timeout)
        # Waypoint advance at WP_RATE_HZ
        self._wp_timer      = self.create_timer(1.0 / WP_RATE_HZ, self._advance_waypoint)

        self.get_logger().info(
            f'CircleSearch ready | timeout={TIMEOUT_SEC}s | '
            f'radius={RADIUS_M}m | steps={NUM_STEPS}'
        )

    # =========================================================================
    #  Callbacks
    # =========================================================================

    def _cb_pose(self, msg: PoseStamped):
        self._current_pose = msg

    def _cb_enabled(self, msg: Bool):
        """Mission planner tells us whether tracking mode is active."""
        was_enabled = self._enabled
        self._enabled = msg.data

        if self._enabled and not was_enabled:
            # Just entered tracking mode — reset the timeout clock
            self._last_det_time = self.get_clock().now()
            self.get_logger().info('CircleSearch: tracking enabled, timeout clock started')

        if not self._enabled and self._searching:
            # Tracking mode switched off externally (e.g. landing) — cancel
            self._cancel_search('tracking disabled externally')

    def _cb_detections(self, msg: DetectionArray):
        """Update the last-seen timestamp whenever the tracker finds the dot."""
        if msg.detections:
            self._last_det_time = self.get_clock().now()

            if self._searching:
                # Dot re-acquired during circle — abort search immediately
                self._cancel_search('red dot re-acquired')

    # =========================================================================
    #  Timeout check
    # =========================================================================

    def _check_timeout(self):
        """Trigger circle search if no detection for TIMEOUT_SEC."""
        if not self._enabled or self._searching:
            return
        if self._current_pose is None:
            return
        if self._last_det_time is None:
            # Haven't seen any detection since enabled; start the clock now
            self._last_det_time = self.get_clock().now()
            return

        elapsed = (self.get_clock().now() - self._last_det_time).nanoseconds * 1e-9
        if elapsed >= TIMEOUT_SEC:
            self._start_search()

    # =========================================================================
    #  Circle search state machine
    # =========================================================================

    def _start_search(self):
        """Begin the circle search pattern."""
        pos = self._current_pose.pose.position
        self._circle_centre = (pos.x, pos.y, pos.z)
        self._step_index    = 0
        self._searching     = True

        self.get_logger().info(
            f'CircleSearch: starting circle at '
            f'({pos.x:.2f}, {pos.y:.2f}, {pos.z:.2f})'
        )

        self._publish_active(True)
        self._publish_current_waypoint()

    def _advance_waypoint(self):
        """Publish current circle waypoint; advance when drone is close enough."""
        if not self._searching or self._current_pose is None:
            return

        # Check whether we've reached the current waypoint
        wp = self._circle_waypoint(self._step_index)
        dx = self._current_pose.pose.position.x - wp.pose.position.x
        dy = self._current_pose.pose.position.y - wp.pose.position.y
        dist = math.hypot(dx, dy)

        if dist <= STEP_ACCEPT_M:
            self._step_index += 1

            if self._step_index >= NUM_STEPS:
                # Full circle complete
                self._searching = False
                self.get_logger().info('CircleSearch: circle complete, returning to hover')
                self._publish_active(False)
                return

            self._publish_current_waypoint()
        else:
            # Re-publish current waypoint so mission_planner keeps targeting it
            self._publish_current_waypoint()

    def _cancel_search(self, reason: str):
        """Abort the circle search early."""
        self._searching  = False
        self._step_index = 0
        self.get_logger().info(f'CircleSearch: cancelled ({reason})')
        self._publish_active(False)

    # =========================================================================
    #  Helpers
    # =========================================================================

    def _circle_waypoint(self, index: int) -> PoseStamped:
        """Return the PoseStamped for waypoint at *index* around the circle."""
        cx, cy, cz = self._circle_centre
        angle = 2.0 * math.pi * index / NUM_STEPS
        x = cx + RADIUS_M * math.cos(angle)
        y = cy + RADIUS_M * math.sin(angle)

        wp = PoseStamped()
        wp.header.stamp    = self.get_clock().now().to_msg()
        wp.header.frame_id = 'map'
        wp.pose.position.x = x
        wp.pose.position.y = y
        wp.pose.position.z = cz
        wp.pose.orientation.w = 1.0
        return wp

    def _publish_current_waypoint(self):
        if self._circle_centre is None:
            return
        wp = self._circle_waypoint(self._step_index)
        self._target_pub.publish(wp)

    def _publish_active(self, value: bool):
        msg = Bool()
        msg.data = value
        self._active_pub.publish(msg)


def main(args=None):
    node = None
    try:
        rclpy.init(args=args)
        node = CircleSearchNode()
        rclpy.spin(node)
    except (KeyboardInterrupt, RuntimeError):
        pass
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
