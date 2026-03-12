#!/usr/bin/env python3

import math
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import TimeReference
from flight_exercise_3.srv import SetPoseSource
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy


def quat_multiply(q1, q2):
    """Multiply two quaternions (x, y, z, w)."""
    return (
        q1[3]*q2[0] + q1[0]*q2[3] + q1[1]*q2[2] - q1[2]*q2[1],
        q1[3]*q2[1] - q1[0]*q2[2] + q1[1]*q2[3] + q1[2]*q2[0],
        q1[3]*q2[2] + q1[0]*q2[1] - q1[1]*q2[0] + q1[2]*q2[3],
        q1[3]*q2[3] - q1[0]*q2[0] - q1[1]*q2[1] - q1[2]*q2[2],
    )

def quat_conjugate(q):
    """Conjugate (inverse for unit quaternions) of (x, y, z, w)."""
    return (-q[0], -q[1], -q[2], q[3])

def quat_rotate_vec(q, v):
    """Rotate a 3-vector by a unit quaternion (x, y, z, w)."""
    vq = (v[0], v[1], v[2], 0.0)
    return quat_multiply(quat_multiply(q, vq), quat_conjugate(q))[:3]


class OdomRepublisher(Node):
    def __init__(self):
        super().__init__('odom_republisher')
        
        # Current pose source: 'vicon', 'realsense', or 'realsense_corrected'
        self.pose_source = 'realsense'
        
        # Topic definitions
        self.vicon_topic = '/vicon/ROB498_Drone/ROB498_Drone'
        self.realsense_topic = '/camera/pose/sample'
        self.mavros_topic = '/mavros/vision_pose/pose'
        self.rs_pub_topic = '/t11/rs'
        self.vicon_pub_topic = '/t11/vicon'

        self.time = self.get_clock().now().to_msg()

        # Latest transformed PoseStamped messages for delta computation and offset calibration
        self.latest_vicon_msg = None    # transformed vicon PoseStamped
        self.latest_realsense_msg = None  # transformed realsense PoseStamped

        # Staleness threshold (seconds) — both must be fresher than this
        self.staleness_thresh = 2.0
        # Sync threshold (seconds) — vicon and realsense stamps must be within this
        self.sync_thresh = 0.05

        # Offset for realsense_corrected mode
        # offset = T_vicon * T_realsense^{-1}, so corrected = offset * realsense
        self.offset_position = None   # (x, y, z)
        self.offset_orientation = None  # (x, y, z, w)
        self.latest_corrected_pose = None  # latest corrected PoseStamped
        
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

        self.realsense_pub = self.create_publisher(
            PoseStamped,
            self.rs_pub_topic,
            qos_profile_r
        )

        self.vicon_pub = self.create_publisher(
            PoseStamped,
            self.vicon_pub_topic,
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

        self.time_sub = self.create_subscription(
            TimeReference,
            "/mavros/time_reference",
            self.time_callback,
            qos_profile
        )

        # 0.2 Hz timer to log vicon-realsense delta
        self.delta_timer = self.create_timer(5.0, self.delta_timer_callback)  # 0.2 Hz
        
        # Service to switch pose source
        self.switch_srv = self.create_service(
            SetPoseSource,
            'odom_republisher/set_pose_source',
            self.switch_source_callback
        )
        
        self.get_logger().info(f'Odom republisher started. Current source: {self.pose_source}')
    
    def switch_source_callback(self, request, response):
        """Service callback to switch between vicon, realsense, and realsense_corrected"""
        valid_sources = ('vicon', 'realsense', 'realsense_corrected')
        if request.source in valid_sources:
            self.pose_source = request.source
        else:
            self.get_logger().error(f'Invalid pose source: {request.source}')
            response.success = False
            return response
        
        self.get_logger().info(f'Switched pose source to: {self.pose_source}')
        response.success = True
        return response
    
    def vicon_callback(self, msg: PoseStamped):
        current_time = self.get_clock().now().to_msg()

        # Build transformed vicon pose (single place to update the transform)
        pub_msg = PoseStamped()
        pub_msg.header.stamp = current_time
        pub_msg.header.frame_id = "map"
        pub_msg.pose = msg.pose
        # pub_msg.pose.position.x = -msg.pose.position.x
        # pub_msg.pose.position.y = -msg.pose.position.y
        # pub_msg.pose.position.z = msg.pose.position.z

        # Store transformed vicon message
        self.latest_vicon_msg = pub_msg

        self.vicon_pub.publish(pub_msg)

        # Update realsense_corrected offset whenever both sources are fresh
        if self.latest_realsense_msg is not None:
            now = self.get_clock().now()
            rs_age = (now - rclpy.time.Time.from_msg(self.latest_realsense_msg.header.stamp)).nanoseconds / 1e9
            if rs_age < self.staleness_thresh:
                vp = pub_msg.pose.position
                vq = pub_msg.pose.orientation
                rp = self.latest_realsense_msg.pose.position
                rq = self.latest_realsense_msg.pose.orientation

                # offset_q = q_vicon * q_realsense^{-1}
                rq_tup = (rq.x, rq.y, rq.z, rq.w)
                vq_tup = (vq.x, vq.y, vq.z, vq.w)
                self.offset_orientation = quat_multiply(vq_tup, quat_conjugate(rq_tup))

                # offset_pos = vicon_pos - R_offset * realsense_pos
                rotated_rs = quat_rotate_vec(self.offset_orientation, (rp.x, rp.y, rp.z))
                self.offset_position = (
                    vp.x - rotated_rs[0],
                    vp.y - rotated_rs[1],
                    vp.z - rotated_rs[2],
                )

        if self.pose_source == 'vicon':
            self.pose_pub.publish(pub_msg)
    
    def realsense_callback(self, msg: Odometry):
        current_time = self.get_clock().now().to_msg()

        # Build transformed realsense pose (single place to update the transform)
        pub_msg = PoseStamped()
        pub_msg.header.stamp = current_time
        pub_msg.header.frame_id = "map"
        pub_msg.pose = msg.pose.pose
        # pub_msg.pose.position.x = -msg.pose.pose.position.x
        # pub_msg.pose.position.y = -msg.pose.pose.position.y
        pub_msg.pose.position.z = msg.pose.pose.position.z - 0.1

        # Store transformed realsense message
        self.latest_realsense_msg = pub_msg

        self.realsense_pub.publish(pub_msg)

        if self.pose_source == 'realsense':
            self.pose_pub.publish(pub_msg)

        # Always compute and store the corrected pose when offset is available
        if self.offset_position is not None:
            rp = pub_msg.pose.position
            rq = pub_msg.pose.orientation
            rq_tup = (rq.x, rq.y, rq.z, rq.w)

            # corrected_q = offset_q * realsense_q
            cq = quat_multiply(self.offset_orientation, rq_tup)

            # corrected_pos = offset_pos + R_offset * realsense_pos
            rotated = quat_rotate_vec(self.offset_orientation, (rp.x, rp.y, rp.z))

            corrected_msg = PoseStamped()
            corrected_msg.header = pub_msg.header
            corrected_msg.pose.position.x = self.offset_position[0] + rotated[0]
            corrected_msg.pose.position.y = self.offset_position[1] + rotated[1]
            corrected_msg.pose.position.z = self.offset_position[2] + rotated[2]
            corrected_msg.pose.orientation.x = cq[0]
            corrected_msg.pose.orientation.y = cq[1]
            corrected_msg.pose.orientation.z = cq[2]
            corrected_msg.pose.orientation.w = cq[3]

            self.latest_corrected_pose = corrected_msg

        if self.pose_source == 'realsense_corrected':
            if self.latest_corrected_pose is not None:
                self.pose_pub.publish(self.latest_corrected_pose)
            else:
                self.pose_pub.publish(pub_msg)  # fallback to raw

    def delta_timer_callback(self):
        """Log the positional and angular deviation between vicon and realsense if both are fresh and in sync."""
        now = self.get_clock().now()

        if self.latest_vicon_msg is None or self.latest_realsense_msg is None:
            return

        vicon_time = rclpy.time.Time.from_msg(self.latest_vicon_msg.header.stamp)
        realsense_time = rclpy.time.Time.from_msg(self.latest_realsense_msg.header.stamp)

        vicon_age = (now - vicon_time).nanoseconds / 1e9
        realsense_age = (now - realsense_time).nanoseconds / 1e9

        if vicon_age > self.staleness_thresh or realsense_age > self.staleness_thresh:
            return

        # Ensure the two measurements are from roughly the same instant
        stamp_diff = abs((vicon_time - realsense_time).nanoseconds) / 1e9
        if stamp_diff > self.sync_thresh:
            return

        # Positional deviation (both stored msgs are already transformed)
        vp = self.latest_vicon_msg.pose.position
        rp = self.latest_realsense_msg.pose.position
        pos_dev = math.sqrt((vp.x - rp.x)**2 + (vp.y - rp.y)**2 + (vp.z - rp.z)**2)

        # Angular deviation via quaternion inner product
        vq = self.latest_vicon_msg.pose.orientation
        rq = self.latest_realsense_msg.pose.orientation
        dot = abs(vq.x * rq.x + vq.y * rq.y + vq.z * rq.z + vq.w * rq.w)
        dot = min(dot, 1.0)  # clamp for acos safety
        ang_dev_deg = math.degrees(2.0 * math.acos(dot))

        self.get_logger().info(
            f'Vicon-Realsense deviation: {pos_dev:.3f} m, {ang_dev_deg:.1f} deg'
        )

        # Also log deviation between vicon and realsense_corrected
        if self.latest_corrected_pose is not None:
            cp = self.latest_corrected_pose.pose.position
            cq = self.latest_corrected_pose.pose.orientation

            corr_pos_dev = math.sqrt((vp.x - cp.x)**2 + (vp.y - cp.y)**2 + (vp.z - cp.z)**2)

            corr_dot = abs(vq.x*cq.x + vq.y*cq.y + vq.z*cq.z + vq.w*cq.w)
            corr_dot = min(corr_dot, 1.0)
            corr_ang_dev = math.degrees(2.0 * math.acos(corr_dot))

            self.get_logger().info(
                f'Vicon-Corrected deviation: {corr_pos_dev:.3f} m, {corr_ang_dev:.1f} deg'
            )

    def time_callback(self, msg: TimeReference):
        self.time = msg.time_ref


def main(args=None):
    rclpy.init(args=args)
    node = OdomRepublisher()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
