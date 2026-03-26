#!/usr/bin/env python3
"""
Dot Tracker Node

Drop-in replacement for obj_tracker_node that detects a red dot using HSV
colour masking instead of YOLO. Publishes the same DetectionArray on
/umbrella/detections so waypoint_generator works with zero changes.

Detection algorithm:
  1. Convert BGR → HSV
  2. Mask red pixels across both HSV hue wraps (0-10 and 160-180)
  3. Morphological open then close to remove noise and fill gaps
  4. Find the largest contour above MIN_AREA pixels
  5. Publish its centroid as a Detection with label="red_dot" and
     confidence = contour_area / image_area
"""

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from umbrella.msg import Detection, DetectionArray

# ---------------------------------------------------------------------------
# Configuration — edit these directly
# ---------------------------------------------------------------------------
CAMERA_TOPIC = 'imx/image/raw'

# Red HSV range 1 (hue 0-10)
HSV_LOW_1  = np.array([  0,  50,  40], dtype=np.uint8)
HSV_HIGH_1 = np.array([ 10, 255, 200], dtype=np.uint8)

# Red HSV range 2 (hue 160-180, the other side of the hue wrap)
HSV_LOW_2  = np.array([160,  50,  40], dtype=np.uint8)
HSV_HIGH_2 = np.array([180, 255, 200], dtype=np.uint8)

# Minimum contour area (pixels) to count as a valid detection
MIN_AREA = 40 
# ---------------------------------------------------------------------------


class DotTrackerNode(Node):
    def __init__(self):
        super().__init__('dot_tracker_node')

        self._bridge = CvBridge()

        qos_be = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self._image_sub = self.create_subscription(
            Image, CAMERA_TOPIC, self._cb_image, qos_be
        )
        self._image_pub = self.create_publisher(
            Image, 'umbrella/annotated_img', qos_be
        )
        self._det_pub = self.create_publisher(
            DetectionArray, 'umbrella/detections', 10
        )

        self.get_logger().info(
            f'DotTracker ready | camera={CAMERA_TOPIC} | '
            f'min_area={MIN_AREA}px'
        )

    def _cb_image(self, msg: Image):
        frame = self._bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        h, w = frame.shape[:2]
        image_area = float(h * w)

        # --- Red HSV mask ---
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.bitwise_or(
            cv2.inRange(hsv, HSV_LOW_1, HSV_HIGH_1),
            cv2.inRange(hsv, HSV_LOW_2, HSV_HIGH_2),
        )

        # Morphological clean-up: open removes small noise, close fills gaps
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        # --- Find largest contour ---
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        det_array = DetectionArray()
        det_array.header = msg.header
        annotated = frame.copy()

        best_contour = None
        best_area = 0.0
        for c in contours:
            area = cv2.contourArea(c)
            if area > best_area:
                best_area = area
                best_contour = c

        if best_contour is not None and best_area >= MIN_AREA:
            M = cv2.moments(best_contour)
            if M['m00'] > 0:
                cx = M['m10'] / M['m00']
                cy = M['m01'] / M['m00']

                confidence = min(best_area / image_area, 1.0)

                det = Detection()
                det.x = float(cx - w / 2.0)
                det.y = float(cy - h / 2.0)
                det.label = 'red_dot'
                det.confidence = confidence
                det_array.detections.append(det)

                # Draw contour, centroid, and label on annotated frame
                cv2.drawContours(annotated, [best_contour], -1, (0, 255, 0), 2)
                cv2.circle(annotated, (int(cx), int(cy)), 6, (0, 255, 0), -1)
                cv2.putText(
                    annotated,
                    f'red_dot {confidence:.3f}',
                    (int(cx) + 8, int(cy) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1, cv2.LINE_AA,
                )

                self.get_logger().info(
                    f'Red dot @ px=({cx:.0f},{cy:.0f}) '
                    f'area={best_area:.0f}px conf={confidence:.4f}',
                    throttle_duration_sec=1.0,
                )
            else:
                self.get_logger().debug('Contour found but moments are zero.')
        else:
            self.get_logger().debug(
                f'No red dot detected (best_area={best_area:.0f} < {MIN_AREA})',
                throttle_duration_sec=2.0,
            )

        self._det_pub.publish(det_array)

        ann_msg = self._bridge.cv2_to_imgmsg(annotated, encoding='bgr8')
        ann_msg.header = msg.header
        self._image_pub.publish(ann_msg)


def main(args=None):
    node = None
    try:
        rclpy.init(args=args)
        node = DotTrackerNode()
        rclpy.spin(node)
    except (KeyboardInterrupt, RuntimeError):
        pass
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
