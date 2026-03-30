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
TARGET_Z       = 0.7   # fallback hover height (metres) when no detection is present
M_PER_PX       = 0.007 # metres of drone movement per pixel of dot offset — tune this

# Height calibration: proportional controller that nudges Z up/down to hold
# a constant dot apparent size (confidence = contour_area / image_area).
#   dot looks too small (conf < CONF_TARGET) -> step up
#   dot looks too big   (conf > CONF_TARGET) -> step down
MIN_Z          = 0.5    # hard minimum hover height (metres) -- never go below
MAX_Z          = 0.75   # hard maximum hover height (metres) -- never go above
CONF_TARGET    = 0.005  # desired dot confidence to maintain — tune this
KP_Z           = 5.0    # proportional gain: metres of Z step per unit conf error
MAX_Z_STEP     = 0.05   # maximum Z correction per control cycle (metres)
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

        # Proportional Z controller: nudge current height toward CONF_TARGET.
        # Positive error (dot too small) -> step up; negative -> step down.
        conf_error = CONF_TARGET - best.confidence
        z_step = KP_Z * conf_error
        z_step = max(-MAX_Z_STEP, min(MAX_Z_STEP, z_step))
        current_z = self._current_pose.pose.position.z
        target_z = max(MIN_Z, min(MAX_Z, current_z + z_step))

        waypoint = self._project_to_world(best.x, best.y, target_z)
        self._wp_pub.publish(waypoint)

        self.get_logger().info(
            f'Published waypoint | '
            f'det="{best.label}" conf={best.confidence:.4f} '
            f'err={conf_error:+.4f} z_step={z_step:+.3f} | '
            f'current_z={current_z:.3f} -> target_z={target_z:.3f} | '
            f'xy=({waypoint.pose.position.x:.3f},{waypoint.pose.position.y:.3f})',
            throttle_duration_sec=0.5,
        )

    # ------------------------------------------------------------------
    #  Pixel → world projection
    # ------------------------------------------------------------------

    def _project_to_world(self, px: float, py: float, target_z: float = TARGET_Z) -> PoseStamped:
        """
        Back-project a pixel centre (px, py) in the downward-facing camera
        image to a world-frame PoseStamped.

        Camera frame convention (px/py are centred, (0,0) = middle of frame):
          dot left  (px < 0) → +X waypoint
          dot right (px > 0) → -X waypoint
          dot below (py > 0) → +Y waypoint
          dot above (py < 0) → -Y waypoint

        target_z: desired hover altitude in metres, computed by the proportional
                  Z controller in _cb_detections and clamped to [MIN_Z, MAX_Z].
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
        wp.pose.position.z = target_z
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
