#!/usr/bin/env python3

import rclpy
import cv2

from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from ultralytics import YOLO
  
class ObjTrackerNode(Node):
    def __init__(self):
        super().__init__('imx_camera_driver')

        self._camera_topic = "imx/image/raw"
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        self.image_sub = self.create_subscription(
            Image, self._camera_topic, self._cb_image, qos_profile
        )

        self._detection_model = YOLO("/home/jetson/ROB498-Capstone/src/umbrella/models/yolo26n.pt")
        self._bridge = CvBridge()

    def _cb_image(self, msg: Image):
        frame = self._bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        results = self.model.predict(
            frame,
            verbose=True,
            device='cuda',
            half=True,
            imgsz=320,
        )
        self.get_logger().info("detections: %d", len(results))
        return


def main(args=None):
  rclpy.init(args=args)
  node = ObjTrackerNode()
  rclpy.spin(node)
  node.destroy_node()
  rclpy.shutdown()

if __name__ == '__main__':
  main()