#!/usr/bin/env python3

import math
from typing import Optional

import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from vision_msgs.msg import Detection2DArray

from follower_math import compute_position_step, select_best_person

class PersonFollowerNode(Node):
    """Subscribes to YOLO detections and drone pose, publishes position setpoints."""

    def __init__(self):
        super().__init__('person_follower')

        self.declare_parameter('detections_topic', '/yolo/detections')
        self.declare_parameter('pose_topic', '/mavros/local_position/pose')
        self.declare_parameter('setpoint_topic', '/mavros/setpoint_position/local')
        self.declare_parameter('target_class', 'person')
        self.declare_parameter('image_width', 640)
        self.declare_parameter('image_height', 640)
        self.declare_parameter('hover_height_m', 1.5)
        self.declare_parameter('kp_image_x', 0.6)
        self.declare_parameter('kp_image_y', 0.6)
        self.declare_parameter('max_xy_step_m', 0.15)
        self.declare_parameter('center_deadband', 0.08)
        self.declare_parameter('command_rate_hz', 10.0)
        self.declare_parameter('min_detection_confidence', 0.35)
        self.declare_parameter('search_hold_seconds', 1.0)

        self._target_class = self.get_parameter('target_class').value
        self._img_w = float(self.get_parameter('image_width').value)
        self._img_h = float(self.get_parameter('image_height').value)
        self._hover_height = float(self.get_parameter('hover_height_m').value)
        self._kp_x = float(self.get_parameter('kp_image_x').value)
        self._kp_y = float(self.get_parameter('kp_image_y').value)
        self._max_step = float(self.get_parameter('max_xy_step_m').value)
        self._deadband = float(self.get_parameter('center_deadband').value)
        self._min_conf = float(self.get_parameter('min_detection_confidence').value)
        self._search_hold = float(self.get_parameter('search_hold_seconds').value)

        self._current_pose: Optional[PoseStamped] = None
        self._target_pose = PoseStamped()
        self._target_pose.header.frame_id = 'map'
        self._target_pose.pose.orientation.w = 1.0
        self._last_detection_time = None

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self._pose_sub = self.create_subscription(
            PoseStamped,
            self.get_parameter('pose_topic').value,
            self._cb_pose,
            qos,
        )
        self._det_sub = self.create_subscription(
            Detection2DArray,
            self.get_parameter('detections_topic').value,
            self._cb_detections,
            10,
        )
        self._setpoint_pub = self.create_publisher(
            PoseStamped,
            self.get_parameter('setpoint_topic').value,
            10,
        )

        rate = float(self.get_parameter('command_rate_hz').value)
        self._timer = self.create_timer(1.0 / rate, self._publish_setpoint)

        self.get_logger().info(
            f'Person follower started: target_class={self._target_class}, '
            f'hover={self._hover_height}m, kp=({self._kp_x},{self._kp_y}), '
            f'max_step={self._max_step}m, deadband={self._deadband}'
        )

    # -- callbacks -----------------------------------------------------------

    def _cb_pose(self, msg: PoseStamped):
        self._current_pose = msg
        if math.isclose(self._target_pose.pose.position.z, 0.0, abs_tol=1e-6):
            self._target_pose.pose.position.x = msg.pose.position.x
            self._target_pose.pose.position.y = msg.pose.position.y
            self._target_pose.pose.position.z = self._hover_height

    def _cb_detections(self, msg: Detection2DArray):
        if self._current_pose is None:
            return

        person = select_best_person(
            msg.detections,
            target_class=self._target_class,
            min_confidence=self._min_conf,
        )

        if person is None:
            self._handle_lost_target()
            return

        cx, cy, _w, _h, confidence = person
        step_x, step_y = compute_position_step(
            cx, cy,
            self._img_w, self._img_h,
            self._kp_x, self._kp_y,
            self._deadband, self._max_step,
        )

        self._target_pose.pose.position.x = (
            self._current_pose.pose.position.x + step_x
        )
        self._target_pose.pose.position.y = (
            self._current_pose.pose.position.y + step_y
        )
        self._target_pose.pose.position.z = self._hover_height
        self._target_pose.header.stamp = self.get_clock().now().to_msg()
        self._last_detection_time = self.get_clock().now()

        self.get_logger().info(
            f'Person (conf={confidence:.2f}) step=({step_x:.3f},{step_y:.3f}) '
            f'-> target=({self._target_pose.pose.position.x:.2f},'
            f'{self._target_pose.pose.position.y:.2f},'
            f'{self._target_pose.pose.position.z:.2f})'
        )

    def _handle_lost_target(self):
        if self._last_detection_time is None:
            return
        elapsed = (
            self.get_clock().now() - self._last_detection_time
        ).nanoseconds / 1e9
        if elapsed > self._search_hold:
            self.get_logger().warn(
                'Lost person detection — holding current setpoint.'
            )

    def _publish_setpoint(self):
        if self._current_pose is None:
            return
        self._target_pose.header.stamp = self.get_clock().now().to_msg()
        self._setpoint_pub.publish(self._target_pose)


def main(args=None):
    rclpy.init(args=args)
    node = PersonFollowerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
