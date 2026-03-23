#!/usr/bin/env python3

import numpy as np
import rclpy
import gi

gi.require_version('Gst', '1.0')
from gi.repository import Gst

from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image


def build_pipeline_str(
    capture_width=256,
    capture_height=256,
    display_width=256,
    display_height=256,
    framerate=8,
    flip_method=0,
):
    return (
        f"nvarguscamerasrc ! "
        f"video/x-raw(memory:NVMM),"
        f"width={capture_width},height={capture_height},"
        f"format=NV12,framerate={framerate}/1 ! "
        f"nvvidconv flip-method={flip_method} ! "
        f"video/x-raw,width={display_width},height={display_height},format=BGRx ! "
        f"videoconvert ! "
        f"video/x-raw,format=BGR ! "
        f"appsink name=appsink emit-signals=true max-buffers=1 drop=true"
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
        self._frame = None
        self._read_failures = 0

        Gst.init(None)
        pipeline_str = build_pipeline_str()
        self._pipeline = Gst.parse_launch(pipeline_str)
        self._appsink = self._pipeline.get_by_name('appsink')
        self._appsink.set_property('emit-signals', True)
        self._appsink.connect('new-sample', self._on_new_sample)

        ret = self._pipeline.set_state(Gst.State.PLAYING)
        if ret == Gst.StateChangeReturn.FAILURE:
            self.get_logger().error(
                f"Failed to start GStreamer pipeline.\nPipeline: {pipeline_str}"
            )
        else:
            self.get_logger().info(
                f"Initialized IMX camera driver (GStreamer backend). Pipeline: {pipeline_str}"
            )

        self._timer = self.create_timer(0.2, self._pub_timer_callback)

    @staticmethod
    def _numpy_to_imgmsg(frame: np.ndarray) -> Image:
        """Convert a BGR numpy array to a sensor_msgs/Image without cv_bridge."""
        msg = Image()
        msg.height, msg.width = frame.shape[:2]
        msg.encoding = 'bgr8'
        msg.is_bigendian = False
        msg.step = frame.shape[1] * 3
        msg.data = frame.tobytes()
        return msg

    def _on_new_sample(self, appsink):
        """Called from GStreamer streaming thread when a new frame arrives."""
        sample = appsink.emit('pull-sample')
        if sample is None:
            return Gst.FlowReturn.ERROR

        buf = sample.get_buffer()
        caps = sample.get_caps()
        structure = caps.get_structure(0)
        height = structure.get_value('height')
        width = structure.get_value('width')

        success, map_info = buf.map(Gst.MapFlags.READ)
        if not success:
            return Gst.FlowReturn.ERROR

        frame = np.frombuffer(map_info.data, dtype=np.uint8).reshape(height, width, 3)
        self._frame = frame.copy()
        buf.unmap(map_info)
        return Gst.FlowReturn.OK

    def _pub_timer_callback(self):
        if self._frame is not None:
            if self._read_failures > 0:
                self.get_logger().info("Camera is now receiving frames.")
            self._read_failures = 0
            self._publisher.publish(self._numpy_to_imgmsg(self._frame))
        else:
            self._read_failures += 1
            if self._read_failures == 1 or self._read_failures % 50 == 0:
                self.get_logger().warn(
                    f"No frame received yet (count={self._read_failures}). "
                    "Check camera connection and nvargus-daemon."
                )

    def destroy_node(self):
        self._pipeline.set_state(Gst.State.NULL)
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = ImxCameraDriver()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
