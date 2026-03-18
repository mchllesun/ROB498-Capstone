#!/usr/bin/env python3

import math
from typing import Optional

import rclpy
import cv2
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image

from ultralytics import YOLO


class PersonFollowYolo(Node):
    """Prototype controller that keeps a detected person centered below the drone."""

    def __init__(self):
        super().__init__('person_follow_yolo')

        self.declare_parameter('image_topic', '/imx/image/raw')
        self.declare_parameter('pose_topic', '/mavros/local_position/pose')
        self.declare_parameter('setpoint_topic', '/mavros/setpoint_position/local')
        self.declare_parameter('model_path', 'yolo26n.pt')
        self.declare_parameter('target_class', 'person')
        self.declare_parameter('use_fp16', True)
        self.declare_parameter('device', 'cpu')
        self.declare_parameter('imgsz', 320)
        self.declare_parameter('hover_height_m', 1.5)
        self.declare_parameter('kp_image_x', 0.6)
        self.declare_parameter('kp_image_y', 0.6)
        self.declare_parameter('max_xy_step_m', 0.15)
        self.declare_parameter('center_deadband', 0.08)
        self.declare_parameter('command_rate_hz', 10.0)
        self.declare_parameter('min_detection_confidence', 0.35)
        self.declare_parameter('search_hold_seconds', 1.0)

        self.image_topic = self.get_parameter('image_topic').value
        self.pose_topic = self.get_parameter('pose_topic').value
        self.setpoint_topic = self.get_parameter('setpoint_topic').value
        self.target_class = self.get_parameter('target_class').value
        self.use_fp16 = bool(self.get_parameter('use_fp16').value)
        self.device = self.get_parameter('device').value
        self.imgsz = int(self.get_parameter('imgsz').value)
        self.hover_height_m = float(self.get_parameter('hover_height_m').value)
        self.kp_image_x = float(self.get_parameter('kp_image_x').value)
        self.kp_image_y = float(self.get_parameter('kp_image_y').value)
        self.max_xy_step_m = float(self.get_parameter('max_xy_step_m').value)
        self.center_deadband = float(self.get_parameter('center_deadband').value)
        self.min_detection_confidence = float(
            self.get_parameter('min_detection_confidence').value
        )
        self.search_hold_seconds = float(self.get_parameter('search_hold_seconds').value)


        model_path = self.get_parameter('model_path').value
        self.model = YOLO(model_path)
        self.bridge = CvBridge()
        self.current_pose: Optional[PoseStamped] = None
        self.target_pose = PoseStamped()
        self.target_pose.header.frame_id = 'map'
        self.target_pose.pose.orientation.w = 1.0

        self.last_detection_time = None

        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self.pose_sub = self.create_subscription(
            PoseStamped, self.pose_topic, self._cb_pose, qos_profile
        )
        self.image_sub = self.create_subscription(
            Image, self.image_topic, self._cb_image, qos_profile
        )
        self.setpoint_pub = self.create_publisher(PoseStamped, self.setpoint_topic, 10)

        command_rate_hz = float(self.get_parameter('command_rate_hz').value)
        self.timer = self.create_timer(1.0 / command_rate_hz, self._publish_setpoint)

        self.get_logger().info(
            f'Person follow node started. image_topic={self.image_topic}, '
            f'pose_topic={self.pose_topic}, model={model_path}, '
            f'device={self.device}, fp16={self.use_fp16}, imgsz={self.imgsz}'
        )

    def _cb_pose(self, msg: PoseStamped):
        self.current_pose = msg
        if self.target_pose.header.frame_id == 'map' and math.isclose(
            self.target_pose.pose.position.z, 0.0, abs_tol=1e-6
        ):
            self.target_pose.pose.position.x = msg.pose.position.x
            self.target_pose.pose.position.y = msg.pose.position.y
            self.target_pose.pose.position.z = self.hover_height_m

    def _cb_image(self, msg: Image):
        if self.current_pose is None:
            return

        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        results = self.model.predict(
            frame,
            verbose=False,
            device=self.device,
            half=self.use_fp16,
            imgsz=self.imgsz,
        )
        if not results:
            self._handle_missing_detection()
            return

        detection = self._select_person(results[0])
        if detection is None:
            self._handle_missing_detection()
            return

        img_h, img_w = frame.shape[:2]
        center_x, center_y, confidence = detection
        norm_err_x = (center_x - (img_w / 2.0)) / (img_w / 2.0)
        norm_err_y = (center_y - (img_h / 2.0)) / (img_h / 2.0)

        step_x = 0.0 if abs(norm_err_y) < self.center_deadband else norm_err_y * self.kp_image_x
        step_y = 0.0 if abs(norm_err_x) < self.center_deadband else norm_err_x * self.kp_image_y

        step_x = max(-self.max_xy_step_m, min(self.max_xy_step_m, step_x))
        step_y = max(-self.max_xy_step_m, min(self.max_xy_step_m, step_y))

        self.target_pose.pose.position.x = self.current_pose.pose.position.x + step_x
        self.target_pose.pose.position.y = self.current_pose.pose.position.y + step_y
        self.target_pose.pose.position.z = self.hover_height_m
        self.target_pose.header.stamp = self.get_clock().now().to_msg()
        self.last_detection_time = self.get_clock().now()

        self.get_logger().debug(
            f'Person detected with confidence {confidence:.2f}. '
            f'Image error=({norm_err_x:.2f}, {norm_err_y:.2f}), '
            f'target=({self.target_pose.pose.position.x:.2f}, '
            f'{self.target_pose.pose.position.y:.2f}, '
            f'{self.target_pose.pose.position.z:.2f})'
        )

    def _select_person(self, result):
        if result.boxes is None or len(result.boxes) == 0:
            return None

        best_detection = None
        best_area = -1.0
        class_names = result.names

        for box in result.boxes:
            cls_idx = int(box.cls.item())
            confidence = float(box.conf.item())
            if confidence < self.min_detection_confidence:
                continue

            if isinstance(class_names, dict):
                class_name = class_names.get(cls_idx, str(cls_idx))
            elif cls_idx < len(class_names):
                class_name = class_names[cls_idx]
            else:
                class_name = str(cls_idx)

            if class_name != self.target_class:
                continue

            x1, y1, x2, y2 = box.xyxy[0].tolist()
            area = max(0.0, (x2 - x1) * (y2 - y1))
            if area <= best_area:
                continue

            best_area = area
            best_detection = ((x1 + x2) / 2.0, (y1 + y2) / 2.0, confidence)

        return best_detection

    def _handle_missing_detection(self):
        if self.last_detection_time is None:
            return

        elapsed = (self.get_clock().now() - self.last_detection_time).nanoseconds / 1e9
        if elapsed > self.search_hold_seconds:
            self.get_logger().warn(
                'Lost person detection. Holding current setpoint until target returns.'
            )

    def _publish_setpoint(self):
        if self.current_pose is None:
            return

        self.target_pose.header.stamp = self.get_clock().now().to_msg()
        self.setpoint_pub.publish(self.target_pose)


def main(args=None):
    rclpy.init(args=args)
    node = PersonFollowYolo()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
