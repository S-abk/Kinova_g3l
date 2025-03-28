#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
import numpy as np
import cv2
import cv2.aruco as aruco
import pyrealsense2 as rs
import threading
import time
import sys
import os

# Kinova Kortex API Imports
from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import BaseCyclicClient
from kortex_api.autogen.messages import Base_pb2, BaseCyclic_pb2, Common_pb2

# Utilities for connection (assumed to be provided)
import utilities

##############################################################################
# Forward Kinematics using classical DH parameters for a 6-DOF robot.
##############################################################################
def dh_transform(a, alpha, d, theta):
    """
    Compute the Denavit-Hartenberg transformation matrix.
    
    Parameters:
      a     : link length (m)
      alpha : link twist (radians)
      d     : link offset (m)
      theta : joint angle (radians)
    
    Returns:
      A 4x4 homogeneous transformation matrix.
    """
    return np.array([
        [np.cos(theta), -np.sin(theta)*np.cos(alpha),  np.sin(theta)*np.sin(alpha), a*np.cos(theta)],
        [np.sin(theta),  np.cos(theta)*np.cos(alpha), -np.cos(theta)*np.sin(alpha), a*np.sin(theta)],
        [0,              np.sin(alpha),                np.cos(alpha),               d],
        [0,              0,                            0,                           1]
    ])

def compute_fk(joint_angles):
    """
    Compute the forward kinematics for the robot using classical DH parameters.
    
    Parameters:
      joint_angles: list or numpy array of 6 joint angles in degrees [q1, q2, q3, q4, q5, q6].
    
    Returns:
      A 4x4 homogeneous transformation matrix representing the end-effector pose in the base frame.
    """
    # Convert joint angles to radians.
    q = np.radians(joint_angles)
    
    # DH parameters (based on your table; d-values are sums in mm converted to m):
    # Joint 1: α₁ = π/2, a₁ = 0.0, d₁ = (128.3 + 115.0) mm, θ₁ = q1
    # Joint 2: α₂ = π,   a₂ = 280.0 mm,    d₂ = 0.0,             θ₂ = q2 + π/2
    # Joint 3: α₃ = π/2, a₃ = 0.0,         d₃ = 0.0,             θ₃ = q3 + π/2
    # Joint 4: α₄ = π/2, a₄ = 0.0,         d₄ = (140.0 + 105.0) mm, θ₄ = q4
    # Joint 5: α₅ = π/2, a₅ = 0.0,         d₅ = (28.5 + 28.5) mm,   θ₅ = q5 + π
    # Joint 6: α₆ = π/2, a₆ = 0.0,         d₆ = (105.0 + 130.0) mm, θ₆ = q6 + π/2
    
    d1 = (128.3 + 115.0) / 1000.0   # 243.3 mm
    a2 = 280.0 / 1000.0             # 280.0 mm
    d4 = (140.0 + 105.0) / 1000.0     # 245.0 mm
    d5 = (28.5 + 28.5) / 1000.0       # 57.0 mm
    d6 = (105.0 + 130.0) / 1000.0     # 235.0 mm

    alpha = [np.pi/2, np.pi, np.pi/2, np.pi/2, np.pi/2, np.pi/2]
    a = [0.0, a2, 0.0, 0.0, 0.0, 0.0]
    d = [d1, 0.0, 0.0, d4, d5, d6]
    theta = [
        q[0],
        q[1] + np.pi/2,
        q[2] + np.pi/2,
        q[3],
        q[4] + np.pi,
        q[5] + np.pi/2
    ]
    
    T = np.eye(4)
    for i in range(6):
        A_i = dh_transform(a[i], alpha[i], d[i], theta[i])
        T = T @ A_i
    return T

##############################################################################
# Helper: Convert waypoint tuple to CartesianWaypoint message.
##############################################################################
def populateCartesianCoordinate(waypoint_tuple):
    """
    Convert a tuple (x, y, z, blending_radius, theta_x, theta_y, theta_z)
    into a CartesianWaypoint message.
    """
    waypoint = Base_pb2.CartesianWaypoint()
    waypoint.pose.x = waypoint_tuple[0]
    waypoint.pose.y = waypoint_tuple[1]
    waypoint.pose.z = waypoint_tuple[2]
    waypoint.blending_radius = waypoint_tuple[3]
    waypoint.pose.theta_x = waypoint_tuple[4]
    waypoint.pose.theta_y = waypoint_tuple[5]
    waypoint.pose.theta_z = waypoint_tuple[6]
    waypoint.reference_frame = Base_pb2.CARTESIAN_REFERENCE_FRAME_BASE
    return waypoint

##############################################################################
# ROS2 Node for Hand-Eye Calibration with Real Motion
##############################################################################
class HandEyeCalibrationNode(Node):
    def __init__(self, base):
        super().__init__('hand_eye_calibration_node')
        self.base = base  # BaseClient instance from Kinova API
        
        # Subscribe to joint states.
        self.joint_state_sub = self.create_subscription(
            JointState, '/joint_states', self.joint_state_callback, 10)
        self.current_joint_states = None

        # Set up RealSense.
        self.pipeline, self.profile = self.start_realsense_pipeline()
        self.camera_matrix, self.dist_coeffs = self.get_realsense_intrinsics(self.profile)
        self.marker_length = 0.04  # Marker size (m).

        # Calibration sample storage.
        self.T_BE_samples = []  # Adjusted robot poses.
        self.T_CT_samples = []  # Detected marker poses.
        self.T_offset = self.get_tag_offset_transform()

        # Waypoints (real motion commands).
        self.waypoints = [
         (0.4,  -0.1, 0.2, 0.0, 0.0, 170.0, 0.0),
         (0.41, -0.05, 0.25, 0.0, 0.0, 170.0, 0.0),
         (0.42,  0.0, 0.2, 0.0, 0.0, 170.0, 0.0),
         (0.43,  0.05, 0.25, 0.0, 0.0, 170.0, 0.0),
         (0.44,  0.1, 0.20, 0.0, 0.0, 170.0, 0.0)
         ]
        self.current_waypoint_index = 0

        # Timer for calibration loop.
        self.timer = self.create_timer(2.0, self.calibration_loop)

    def joint_state_callback(self, msg):
        self.current_joint_states = msg.position
        self.get_logger().info(f"Joint states: {msg.position}")

    def start_realsense_pipeline(self):
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        profile = pipeline.start(config)
        return pipeline, profile

    def get_realsense_intrinsics(self, profile):
        video_profile = profile.get_stream(rs.stream.color)
        intr = video_profile.as_video_stream_profile().get_intrinsics()
        camera_matrix = np.array([
            [intr.fx, 0, intr.ppx],
            [0, intr.fy, intr.ppy],
            [0, 0, 1]
        ], dtype=np.float64)
        dist_coeffs = np.array(intr.coeffs, dtype=np.float64).reshape((5, 1))
        return camera_matrix, dist_coeffs

    def get_realsense_image(self):
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            return None
        image = np.asanyarray(color_frame.get_data())
        return image

    def get_tag_offset_transform(self):
        T_offset = np.eye(4, dtype=np.float64)
        T_offset[0, 3] = 0.05  # 5 cm offset along x-axis.
        return T_offset

    def detect_aruco_pose(self, image):
        dictionary = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
        parameters = aruco.DetectorParameters_create()
        corners, ids, _ = aruco.detectMarkers(image, dictionary, parameters=parameters)
        if ids is not None and len(ids) > 0:
            rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(
                corners, self.marker_length, self.camera_matrix, self.dist_coeffs)
            return rvecs[0], tvecs[0]
        else:
            return None, None

    def pose_to_transform(self, rvec, tvec):
        R, _ = cv2.Rodrigues(rvec)
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = tvec.flatten()
        return T

    def execute_waypoint(self, waypoint_tuple):
        """
        Execute a real waypoint motion using the Kinova API.
        """
        waypoint = populateCartesianCoordinate(waypoint_tuple)
        waypoint_list = Base_pb2.WaypointList()
        waypoint_list.duration = 0.0  # immediate execution
        waypoint_list.use_optimal_blending = False
        wp = waypoint_list.waypoints.add()
        wp.name = "waypoint"
        wp.cartesian_waypoint.CopyFrom(waypoint)
        
        # Validate the waypoint list.
        result = self.base.ValidateWaypointList(waypoint_list)
        if len(result.trajectory_error_report.trajectory_error_elements) > 0:
            self.get_logger().error("Waypoint validation failed.")
            return False
        
        # Set up notification callback to wait for motion completion.
        event = threading.Event()
        def callback(notification, event=event):
            self.get_logger().info("EVENT: " + Base_pb2.ActionEvent.Name(notification.action_event))
            if notification.action_event in [Base_pb2.ACTION_END, Base_pb2.ACTION_ABORT]:
                event.set()
        notif_handle = self.base.OnNotificationActionTopic(callback, Base_pb2.NotificationOptions())
        self.get_logger().info("Executing waypoint via real motion...")
        self.base.ExecuteWaypointTrajectory(waypoint_list)
        finished = event.wait(30)  # wait up to 30 seconds
        self.base.Unsubscribe(notif_handle)
        if finished:
            self.get_logger().info("Waypoint reached.")
            return True
        else:
            self.get_logger().error("Timeout during waypoint execution.")
            return False

    def calibration_loop(self):
        if self.current_joint_states is None:
            self.get_logger().warn("No joint state data yet.")
            return

        # Compute end-effector pose using FK.
        T_BE = compute_fk(self.current_joint_states)
        T_BE_adjusted = T_BE @ self.T_offset  # Adjust for tag offset.
        self.T_BE_samples.append(T_BE_adjusted)
        self.get_logger().info("Captured T_BE (adjusted): " + str(T_BE_adjusted))

        # Capture image from RealSense and detect marker.
        image = self.get_realsense_image()
        if image is None:
            self.get_logger().warn("No image captured from RealSense.")
            return
        rvec, tvec = self.detect_aruco_pose(image)
        if rvec is None or tvec is None:
            self.get_logger().warn("ArUco marker not detected.")
            return
        T_CT = self.pose_to_transform(rvec, tvec)
        self.T_CT_samples.append(T_CT)
        self.get_logger().info("Captured T_CT: " + str(T_CT))

        # (Optional) Display the image.
        cv2.imshow("RealSense", image)
        cv2.waitKey(1)

        # Execute real motion to the next waypoint.
        if self.current_waypoint_index < len(self.waypoints):
            waypoint = self.waypoints[self.current_waypoint_index]
            success = self.execute_waypoint(waypoint)
            if success:
                self.current_waypoint_index += 1
        else:
            # If all waypoints have been processed, perform calibration.
            if len(self.T_BE_samples) >= 3 and len(self.T_CT_samples) >= 3:
                self.perform_calibration()
                self.timer.cancel()

    def perform_calibration(self):
        R_gripper2base = []
        t_gripper2base = []
        R_target2cam = []
        t_target2cam = []
        for T_BE, T_CT in zip(self.T_BE_samples, self.T_CT_samples):
            R_gripper2base.append(T_BE[:3, :3])
            t_gripper2base.append(T_BE[:3, 3])
            T_target2cam = np.linalg.inv(T_CT)
            R_target2cam.append(T_target2cam[:3, :3])
            t_target2cam.append(T_target2cam[:3, 3])
        X_rot, X_trans = cv2.calibrateHandEye(
            R_gripper2base, t_gripper2base,
            R_target2cam, t_target2cam,
            method=cv2.CALIB_HAND_EYE_TSAI
        )
        X = np.eye(4, dtype=np.float64)
        X[:3, :3] = X_rot
        X[:3, 3] = X_trans.flatten()
        self.get_logger().info("Hand-Eye Calibration (X): " + str(X))

    def set_servoing_mode(self):
        servo_mode = Base_pb2.ServoingModeInformation()
        servo_mode.servoing_mode = Base_pb2.SINGLE_LEVEL_SERVOING
        self.base.SetServoingMode(servo_mode)
        self.get_logger().info("Servoing mode set to SINGLE_LEVEL_SERVOING.")


    def close_gripper(self):
        """
        Closes the gripper using the Kinova API.
        """
        gripper_command = Base_pb2.GripperCommand()
        finger = gripper_command.gripper.finger.add()
        gripper_command.mode = Base_pb2.GRIPPER_POSITION
        finger.finger_identifier = 1
        finger.value = 1.0  # Adjust this value as needed.
        self.base.SendGripperCommand(gripper_command)
        self.get_logger().info("Gripper closed.")

    def shutdown(self):
        self.pipeline.stop()
        cv2.destroyAllWindows()

##############################################################################
# Main: Integrate ROS2 node with Kinova connection.
##############################################################################
def main(args=None):
    rclpy.init(args=args)
    kinova_args = utilities.parseConnectionArguments()
    with utilities.DeviceConnection.createTcpConnection(kinova_args) as router:
        base = BaseClient(router)
        base_cyclic = BaseCyclicClient(router)  # Not used here but available if needed.
        node = HandEyeCalibrationNode(base)
        
        # Close gripper to secure the attached tag.
        node.set_servoing_mode()
        node.close_gripper()
        print("Place the ArUco marker at the desired offset from the gripper.")
        input("After positioning the marker, press Enter to begin waypoint navigation...")

        try:
            rclpy.spin(node)
        except KeyboardInterrupt:
            pass
        finally:
            node.shutdown()
            node.destroy_node()
            rclpy.shutdown()

if __name__ == '__main__':
    main()
