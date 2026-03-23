#!/usr/bin/env python3

import cv2
import rclpy

from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

def gstreamer_pipeline(
  capture_width=1280,
  capture_height=720,
  display_width=640,
  display_height=640,
  framerate=10,
  flip_method=0,
  ):
  return (
      "nvarguscamerasrc ! "
      "video/x-raw(memory:NVMM),"
      "width=(int)%d, height=(int)%d, "
      "format=(string)NV12, framerate=(fraction)%d/1 ! "
      "nvvidconv flip-method=%d ! "
      "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
      "videoconvert ! "
      "video/x-raw, format=(string)BGR ! appsink"
      % (
          capture_width,
          capture_height,
          framerate,
          flip_method,
          display_width,
          display_height,
      )
  )
  
class ImxCameraDriver(Node):
  def __init__(self):
    super().__init__('imx_camera_driver')
    
    self._camera_topic = "imx/image/raw"
    qos_profile = QoSProfile(
        reliability=ReliabilityPolicy.BEST_EFFORT,
        history=HistoryPolicy.KEEP_LAST,
        depth=10
    )

    self._publisher = self.create_publisher(Image, self._camera_topic, qos_profile)

    self._timer = self.create_timer(0.1, self.pub_timer_callback)

    self._br = CvBridge()
    self._capture = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)

    self.get_logger().info("Initialized IMX camera driver")

  def pub_timer_callback(self):
    ret, frame = self._capture.read()
    if ret:
      self._publisher.publish(self._br.cv2_to_imgmsg(frame))

def main(args=None):
  rclpy.init(args=args)
  node = ImxCameraDriver()
  rclpy.spin(node)
  node.destroy_node()
  rclpy.shutdown()

if __name__ == '__main__':
  main()