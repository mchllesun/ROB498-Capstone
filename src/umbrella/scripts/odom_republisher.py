#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import TimeReference
from umbrella.srv import SetPoseSource
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

class OdomRepublisher(Node):
    def __init__(self):
        super().__init__('odom_republisher')
        
        # Current pose source: 'vicon' or 'realsense'
        self.pose_source = 'realsense'
        
        # Topic definitions
        self.vicon_topic = '/vicon/ROB498_Drone/ROB498_Drone'
        self.realsense_topic = '/camera/pose/sample'
        self.mavros_topic = '/mavros/vision_pose/pose'

        self.time = self.get_clock().now().to_msg()
        
        # QoS profile: sensor poses are best effort
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        qos_profile_r = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        
        # Publisher to mavros vision pose
        self.pose_pub = self.create_publisher(
            PoseStamped,
            self.mavros_topic,
            qos_profile_r
        )

        self.vicon_sub = self.create_subscription(
            PoseStamped,
            self.vicon_topic,
            self.vicon_callback,
            qos_profile
        )
        
        self.realsense_sub = self.create_subscription(
            Odometry,
            self.realsense_topic,
            self.realsense_callback,
            qos_profile
        )
        
        # Service to switch pose source
        self.switch_srv = self.create_service(
            SetPoseSource,
            'odom_republisher/set_pose_source',
            self.switch_source_callback
        )
        
        self.get_logger().info(f'Odom republisher started. Current source: {self.pose_source}')
    
    def switch_source_callback(self, request, response):
        """Service callback to switch between vicon and realsense"""
        if request.source == 'vicon':
            self.pose_source = 'vicon'
        elif request.source == 'realsense':
            self.pose_source = 'realsense'
        else:
            self.get_logger().error(f'Invalid pose source: {request.source}')
            response.success = False
            return response
        
        self.get_logger().info(f'Switched pose source to: {self.pose_source}')
        response.success = True
        return response
    
    def vicon_callback(self, msg: PoseStamped):
        current_time = self.get_clock().now().to_msg()
        if self.pose_source == 'vicon':
            pub_msg = PoseStamped()
            pub_msg.header.stamp = current_time
            pub_msg.header.frame_id = "map"
            pub_msg.pose = msg.pose

            pub_msg.pose.position.z = msg.pose.position.z + 0.1

            self.pose_pub.publish(pub_msg)
    
    def realsense_callback(self, msg: Odometry):
        current_time = self.get_clock().now().to_msg()
        if self.pose_source == 'realsense':
            pub_msg = PoseStamped()
            pub_msg.header.stamp = current_time
            pub_msg.header.frame_id = "map"
            pub_msg.pose = msg.pose.pose

            self.pose_pub.publish(pub_msg)


def main(args=None):
    rclpy.init(args=args)
    node = OdomRepublisher()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
