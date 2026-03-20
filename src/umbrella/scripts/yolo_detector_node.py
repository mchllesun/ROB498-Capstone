#!/usr/bin/env python3

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from vision_msgs.msg import (
    Detection2D,
    Detection2DArray,
    ObjectHypothesisWithPose,
)

from ultralytics import YOLO


class YoloDetectorNode(Node):
    """Runs YOLO inference on camera images and publishes detections."""

    def __init__(self):
        super().__init__('yolo_detector')

        self.declare_parameter('image_topic', '/imx/image/raw')
        self.declare_parameter('detections_topic', '/yolo/detections')
        self.declare_parameter('model_path', 'yolo26n.pt')
        self.declare_parameter('device', 'cuda:0')
        self.declare_parameter('use_fp16', False)
        self.declare_parameter('imgsz', 320)
        self.declare_parameter('conf_threshold', 0.35)

        image_topic = self.get_parameter('image_topic').value
        detections_topic = self.get_parameter('detections_topic').value
        model_path = self.get_parameter('model_path').value
        self._device = self.get_parameter('device').value
        self._use_fp16 = bool(self.get_parameter('use_fp16').value)
        self._imgsz = int(self.get_parameter('imgsz').value)
        self._conf_threshold = float(self.get_parameter('conf_threshold').value)

        self._model = YOLO(model_path)
        self._frame_count = 0

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self._image_sub = self.create_subscription(
            Image, image_topic, self._cb_image, qos
        )
        self._det_pub = self.create_publisher(Detection2DArray, detections_topic, 10)

        self.get_logger().info(
            f'YOLO detector started: topic={image_topic}, model={model_path}, '
            f'device={self._device}, fp16={self._use_fp16}, imgsz={self._imgsz}'
        )

    def _cb_image(self, msg: Image):
        frame = np.frombuffer(msg.data, dtype=np.uint8).reshape(
            msg.height, msg.width, 3
        )

        results = self._model.predict(
            frame,
            verbose=False,
            device=self._device,
            half=self._use_fp16,
            imgsz=self._imgsz,
            conf=self._conf_threshold,
        )

        det_array = Detection2DArray()
        det_array.header = msg.header

        if results and results[0].boxes is not None:
            class_names = results[0].names
            for box in results[0].boxes:
                det = Detection2D()
                det.header = msg.header

                x1, y1, x2, y2 = box.xyxy[0].tolist()
                det.bbox.center.position.x = (x1 + x2) / 2.0
                det.bbox.center.position.y = (y1 + y2) / 2.0
                det.bbox.size_x = float(x2 - x1)
                det.bbox.size_y = float(y2 - y1)

                hyp = ObjectHypothesisWithPose()
                cls_idx = int(box.cls.item())
                if isinstance(class_names, dict):
                    hyp.hypothesis.class_id = class_names.get(cls_idx, str(cls_idx))
                elif cls_idx < len(class_names):
                    hyp.hypothesis.class_id = class_names[cls_idx]
                else:
                    hyp.hypothesis.class_id = str(cls_idx)
                hyp.hypothesis.score = float(box.conf.item())
                det.results.append(hyp)

                det_array.detections.append(det)

        self._det_pub.publish(det_array)

        self._frame_count += 1
        if self._frame_count % 30 == 0:
            n = len(det_array.detections)
            self.get_logger().info(f'Frame {self._frame_count}: {n} detections')


def main(args=None):
    rclpy.init(args=args)
    node = YoloDetectorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
