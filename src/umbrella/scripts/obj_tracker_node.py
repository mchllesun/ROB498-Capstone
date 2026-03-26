#!/usr/bin/env python3

import rclpy
import cv2

from rclpy.node import Node
import numpy as np
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image
from umbrella.msg import Detection, DetectionArray
from cv_bridge import CvBridge

from ultralytics import YOLO
  
class ObjTrackerNode(Node):
    def __init__(self):
        super().__init__('obj_tracker_node')

        self._camera_topic = "imx/image/raw"
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        self.image_sub = self.create_subscription(
            Image, self._camera_topic, self._cb_image, qos_profile
        )

        self.image_pub = self.create_publisher(
            Image, "umbrella/annotated_img", qos_profile
        )

        self.detection_pub = self.create_publisher(
            DetectionArray, "umbrella/detections", 10
        )

        self._detection_model = YOLO("/home/jetson/ROB498-Capstone/install/umbrella/share/umbrella/models/yolo26s.pt")
        self._bridge = CvBridge()

        self.get_logger().info("Running warmup inference (may take 10-30s)...")
        dummy = np.zeros((256, 256, 3), dtype=np.uint8)
        self._detection_model.predict(
            dummy,
            verbose=False,
            device='cuda',
            half=True,
            imgsz=256,
            conf=0.35,
        )
        self.get_logger().info("Warmup done.")

    def _cb_image(self, msg: Image):
        frame = self._bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        results = self._detection_model.predict(
            frame,
            verbose=False,
            device='cuda',
            half=True,
            imgsz=256,
            conf=0.35,
        )

        self.get_logger().info(f"Detections: {len(results[0].boxes)}")

        # Publish annotated image
        annotated_frame = results[0].plot()
        out_msg = self._bridge.cv2_to_imgmsg(annotated_frame, encoding='bgr8')
        out_msg.header = msg.header
        self.image_pub.publish(out_msg)

        # Publish detections array
        detection_array = DetectionArray()
        detection_array.header = msg.header

        if results and results[0].boxes is not None:
            for box in results[0].boxes:
                det = Detection()
                # xywh is center_x, center_y, width, height
                xywh = box.xywh[0]
                det.x = float(xywh[0])
                det.y = float(xywh[1])
                
                cls_id = int(box.cls[0])
                det.label = results[0].names[cls_id]
                det.confidence = float(box.conf[0])
                
                detection_array.detections.append(det)

        self.detection_pub.publish(detection_array)
        return


def main(args=None):
  rclpy.init(args=args)
  node = ObjTrackerNode()
  rclpy.spin(node)
  node.destroy_node()
  rclpy.shutdown()

if __name__ == '__main__':
  main()