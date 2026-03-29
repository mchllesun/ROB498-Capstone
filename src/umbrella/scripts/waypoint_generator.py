#!/usr/bin/env python3
"""
Waypoint Generator Node

Subscribes to:
  - drone pose          : /mavros/local_position/pose  (PoseStamped)
  - YOLO detections     : /umbrella/detections          (DetectionArray)

Publishes:
  - target waypoint     : /waypoint_generator/target   (PoseStamped)

For each incoming DetectionArray, picks the best detection (highest
confidence, optionally filtered by label) and back-projects its pixel
centre through a pinhole camera model to produce a world-frame XY offset
from the drone's current position. The waypoint is published at the same
altitude (z) as the current drone pose.

Pixel → world transform (downward-facing camera):
  h           = drone altitude (metres)
  ground_width = 2 * h * tan(hfov / 2)
  m_per_px    = ground_width / image_width
  dx_body     = (det.x - cx) * m_per_px   # right in body frame
  dy_body     = (det.y - cy) * m_per_px   # forward in body frame

  Then rotate by drone yaw to get world-frame offsets.
"""

import math
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from umbrella.msg import DetectionArray

# ---------------------------------------------------------------------------
# Configuration — edit these directly
# ---------------------------------------------------------------------------
POSE_TOPIC       = '/mavros/local_position/pose'
DETECTIONS_TOPIC = '/umbrella/detections'
OUTPUT_TOPIC     = '/waypoint_generator/target'

TARGET_LABEL   = ''    # filter to this class label; '' means any class
MIN_CONFIDENCE = 0.0   # dot tracker confidence = contour_area/image_area, typically ~0.001-0.05
TARGET_Z       = 0.4   # fixed hover height (metres) for all published waypoints
M_PER_PX       = 0.003 # metres of drone movement per pixel of dot offset — tune this
# ---------------------------------------------------------------------------


def _quat_to_yaw(q) -> float:
    """Extract yaw (rotation about Z) from a geometry_msgs/Quaternion."""
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


class WaypointGeneratorNode(Node):
    def __init__(self):
        super().__init__('waypoint_generator')

        self._target_label = TARGET_LABEL
        self._min_conf     = MIN_CONFIDENCE

        # Default pose: origin, facing forward, at 1m altitude
        default_pose = PoseStamped()
        default_pose.header.frame_id = 'map'
        default_pose.pose.position.z = 1.0
        default_pose.pose.orientation.w = 1.0  # identity quaternion (yaw=0)
        self._current_pose: PoseStamped = default_pose

        # --- QoS ---
        be_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        rel_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        # --- Subscribers ---
        self._pose_sub = self.create_subscription(
            PoseStamped, POSE_TOPIC, self._cb_pose, be_qos
        )
        self._det_sub = self.create_subscription(
            DetectionArray, DETECTIONS_TOPIC, self._cb_detections, rel_qos
        )

        # --- Publisher ---
        self._wp_pub = self.create_publisher(PoseStamped, OUTPUT_TOPIC, 10)

        self.get_logger().info(
            f'WaypointGenerator ready | '
            f'pose={POSE_TOPIC} | detections={DETECTIONS_TOPIC} | '
            f'output={OUTPUT_TOPIC} | '
            f'label="{self._target_label or "any"}" | '
            f'm_per_px={M_PER_PX}'
        )

    # ------------------------------------------------------------------
    #  Callbacks
    # ------------------------------------------------------------------

    def _cb_pose(self, msg: PoseStamped):
        self._current_pose = msg

    def _cb_detections(self, msg: DetectionArray):
        if not msg.detections:
            return

        # Pick best detection
        candidates = msg.detections
        if self._target_label:
            candidates = [d for d in candidates if d.label == self._target_label]
        candidates = [d for d in candidates if d.confidence >= self._min_conf]

        if not candidates:
            self.get_logger().debug(
                'Detections received but none passed label/confidence filter.',
                throttle_duration_sec=2.0,
            )
            return

        best = max(candidates, key=lambda d: d.confidence)
        waypoint = self._project_to_world(best.x, best.y)
        self._wp_pub.publish(waypoint)

        self.get_logger().info(
            f'Published waypoint | '
            f'det="{best.label}" ({best.confidence:.2f}) '
            f'px=({best.x:.0f},{best.y:.0f}) | '
            f'xyz=({waypoint.pose.position.x:.3f}, '
            f'{waypoint.pose.position.y:.3f}, '
            f'{waypoint.pose.position.z:.3f})'
        )

    # ------------------------------------------------------------------
    #  Pixel → world projection
    # ------------------------------------------------------------------

    def _project_to_world(self, px: float, py: float) -> PoseStamped:
        """
        Back-project a pixel centre (px, py) in the downward-facing camera
        image to a world-frame PoseStamped at the drone's current altitude.

        Camera frame convention (px/py are centred, (0,0) = middle of frame):
          dot left  (px < 0) → +X waypoint
          dot right (px > 0) → -X waypoint
          dot below (py > 0) → +Y waypoint
          dot above (py < 0) → -Y waypoint
        """
        # Offset from drone in body frame — axes are independent
        dx_body = -px * M_PER_PX   # left of centre  → +X
        dy_body =  py * M_PER_PX   # below centre    → +Y

        # Rotate body frame → world frame using drone yaw
        yaw = _quat_to_yaw(self._current_pose.pose.orientation)
        dx_world = dx_body * math.cos(yaw) - dy_body * math.sin(yaw)
        dy_world = dx_body * math.sin(yaw) + dy_body * math.cos(yaw)

        wp = PoseStamped()
        wp.header.stamp = self.get_clock().now().to_msg()
        wp.header.frame_id = 'map'
        wp.pose.position.x = self._current_pose.pose.position.x + dx_world
        wp.pose.position.y = self._current_pose.pose.position.y + dy_world
        wp.pose.position.z = TARGET_Z
        # wp.pose.orientation = self._current_pose.pose.orientation  # preserve yaw
        return wp


def main(args=None):
    node = None
    try:
        rclpy.init(args=args)
        node = WaypointGeneratorNode()
        rclpy.spin(node)
    except (KeyboardInterrupt, RuntimeError):
        pass
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
