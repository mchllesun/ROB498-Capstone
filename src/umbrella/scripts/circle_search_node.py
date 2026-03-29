#!/usr/bin/env python3
"""
Spiral Search Node

A dumb waypoint generator that listens for a trigger on `/circle_search/reset`.
When triggered, it captures its current position and begins outputting points
along an expanding spiral to `/circle_search/target`. It does not handle
enable/disable logic or timeouts itself.

Topics
------
Subscribes:
  /mavros/local_position/pose   (PoseStamped)     - drone position
  /circle_search/reset          (Empty)           - trigger to start/restart spiral
Publishes:
  /circle_search/target         (PoseStamped)     - spiral waypoints
"""

import math
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Empty

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SPIRAL_PITCH_M   = 0.50  # distance between outward loops
POINTS_PER_REV   = 36    # angular resolution (every 10°)
STEP_ACCEPT_M    = 0.20  # metres within which a waypoint is considered reached
WP_RATE_HZ       = 5.0   # how often to check whether to advance the waypoint
# ---------------------------------------------------------------------------


class CircleSearchNode(Node):

    def __init__(self):
        super().__init__('circle_search_node')

        self._searching      = False
        self._current_pose   = None

        # Spiral state
        self._spiral_centre  = None   # (x, y, z) tuple captured at search start
        self._step_index     = 0      # current waypoint index

        # QoS
        be_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        # --- Subscribers ---
        self._pose_sub = self.create_subscription(
            PoseStamped, '/mavros/local_position/pose', self._cb_pose, be_qos
        )
        self._reset_sub = self.create_subscription(
            Empty, '/circle_search/reset', self._cb_reset, 10
        )

        # --- Publishers ---
        self._target_pub = self.create_publisher(PoseStamped, '/circle_search/target', 10)

        # --- Timers ---
        # Waypoint advance at WP_RATE_HZ
        self._wp_timer = self.create_timer(1.0 / WP_RATE_HZ, self._advance_waypoint)

        self.get_logger().info(
            f'SpiralSearch ready | pitch={SPIRAL_PITCH_M}m | points/rev={POINTS_PER_REV}'
        )

    def _cb_pose(self, msg: PoseStamped):
        self._current_pose = msg

    def _cb_reset(self, msg: Empty):
        """Trigger or restart the spiral pattern."""
        if self._current_pose is None:
            self.get_logger().warn("Cannot start spiral: no current pose available.")
            return

        pos = self._current_pose.pose.position
        self._spiral_centre = (pos.x, pos.y, pos.z)
        self._step_index    = 0
        self._searching     = True

        self.get_logger().info(
            f'SpiralSearch: starting from '
            f'({pos.x:.2f}, {pos.y:.2f}, {pos.z:.2f})'
        )

        self._publish_current_waypoint()

    def _advance_waypoint(self):
        """Publish current spiral waypoint; advance when drone is close enough."""
        if not self._searching or self._current_pose is None or self._spiral_centre is None:
            return

        # Check whether we've reached the current waypoint
        wp = self._spiral_waypoint(self._step_index)
        dx = self._current_pose.pose.position.x - wp.pose.position.x
        dy = self._current_pose.pose.position.y - wp.pose.position.y
        dist = math.hypot(dx, dy)

        if dist <= STEP_ACCEPT_M:
            self._step_index += 1

        # Always publish the current target so the planner follows it
        self._publish_current_waypoint()

    def _spiral_waypoint(self, index: int) -> PoseStamped:
        """Return the PoseStamped for waypoint at *index* along the spiral."""
        cx, cy, cz = self._spiral_centre
        
        # Archimedian spiral: r = (pitch / 2pi) * theta
        theta = 2.0 * math.pi * index / POINTS_PER_REV
        r = SPIRAL_PITCH_M * (theta / (2.0 * math.pi))
        
        x = cx + r * math.cos(theta)
        y = cy + r * math.sin(theta)

        wp = PoseStamped()
        wp.header.stamp    = self.get_clock().now().to_msg()
        wp.header.frame_id = 'map'
        wp.pose.position.x = x
        wp.pose.position.y = y
        wp.pose.position.z = 0.4
        wp.pose.orientation.w = 1.0
        return wp

    def _publish_current_waypoint(self):
        if not self._searching:
            return
        wp = self._spiral_waypoint(self._step_index)
        self._target_pub.publish(wp)


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
