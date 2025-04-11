import os
import sys
import time
import threading
import numpy as np
import cv2
import pyrealsense2 as rs

# Manipulator API Imports
from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import BaseCyclicClient
from kortex_api.autogen.messages import Base_pb2, BaseCyclic_pb2, Common_pb2

TIMEOUT_DURATION = 30

#######################################
# Helper Functions for Robot Motion
#######################################
def check_for_end_or_abort(e):
    """Closure to monitor action events."""
    def check(notification, e=e):
        print("EVENT: " + Base_pb2.ActionEvent.Name(notification.action_event))
        if (notification.action_event == Base_pb2.ACTION_END or
            notification.action_event == Base_pb2.ACTION_ABORT):
            e.set()
    return check

def populateCartesianCoordinate(waypointInformation):
    """
    Populate a CartesianWaypoint message.
    waypointInformation: (x, y, z, blending_radius, theta_x, theta_y, theta_z)
    """
    waypoint = Base_pb2.CartesianWaypoint()
    waypoint.pose.x = waypointInformation[0]
    waypoint.pose.y = waypointInformation[1]
    waypoint.pose.z = waypointInformation[2]
    waypoint.blending_radius = waypointInformation[3]
    waypoint.pose.theta_x = waypointInformation[4]
    waypoint.pose.theta_y = waypointInformation[5]
    waypoint.pose.theta_z = waypointInformation[6]
    waypoint.reference_frame = Base_pb2.CARTESIAN_REFERENCE_FRAME_BASE
    return waypoint

#######################################
# RealSense & ArUco Detection Functions
#######################################
def build_transform(rvec, tvec):
    """Build a 4x4 homogeneous transformation matrix from rvec and tvec."""
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = R
    T[:3, 3] = tvec.reshape(3)
    return T

def invert_transform(T):
    """Invert a 4x4 homogeneous transformation matrix."""
    R = T[:3, :3]
    t = T[:3, 3]
    R_inv = R.T
    t_inv = -R_inv @ t
    T_inv = np.eye(4, dtype=np.float32)
    T_inv[:3, :3] = R_inv
    T_inv[:3, 3] = t_inv
    return T_inv

#######################################
# 6D Kalman Filter for Fusing 3D Position Measurements
#######################################
class KalmanFilter6D:
    def __init__(self, dt, initial_state):
        # State vector: [x, y, z, vx, vy, vz]^T
        self.x = initial_state  # shape (6,1)
        self.P = np.eye(6) * 1.0

        # State transition matrix (constant velocity model)
        self.A = np.eye(6)
        self.A[0, 3] = dt
        self.A[1, 4] = dt
        self.A[2, 5] = dt

        # Process noise covariance (can tweak as needed)
        self.Q = np.diag([0.01, 0.01, 0.01, 0.1, 0.1, 0.1])

        # Measurement matrix: we measure only position (x, y, z)
        self.H = np.hstack((np.eye(3), np.zeros((3, 3))))

        # Measurement noise covariance for the two sensor types
        self.R_pose = np.eye(3) * 0.05   # Pose estimation (marker detection)
        self.R_depth = np.eye(3) * 0.1     # Direct depth (back-projected) measurement

    def predict(self, dt):
        # Update transition matrix with new dt
        self.A[0, 3] = dt
        self.A[1, 4] = dt
        self.A[2, 5] = dt
        self.x = self.A @ self.x
        self.P = self.A @ self.P @ self.A.T + self.Q

    def update(self, z, R):
        # z is expected as a (3,) vector
        z = z.reshape((3, 1))
        y = z - self.H @ self.x                    # Measurement residual
        S = self.H @ self.P @ self.H.T + R           # Residual covariance
        K = self.P @ self.H.T @ np.linalg.inv(S)     # Kalman gain
        self.x = self.x + K @ y
        self.P = (np.eye(6) - K @ self.H) @ self.P

#######################################
# run_camera_detection(): Using Fused Position Data
#######################################
def run_camera_detection():
    """
    Runs the RealSense pipeline and ArUco detection until the object marker (ID 3)
    is detected and fused over several frames. Returns the fused object's 3D pose
    (position) in the manipulator base frame.
    """
    # Initialize RealSense pipeline
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    profile = pipeline.start(config)
    
    # Get camera intrinsics
    color_stream = profile.get_stream(rs.stream.color)
    color_profile = color_stream.as_video_stream_profile()
    intrinsics = color_profile.get_intrinsics()
    camera_matrix = np.array([[intrinsics.fx, 0, intrinsics.ppx],
                               [0, intrinsics.fy, intrinsics.ppy],
                               [0, 0, 1]])
    dist_coeffs = np.array(intrinsics.coeffs)
    print("Camera matrix:\n", camera_matrix)
    print("Distortion coefficients:\n", dist_coeffs)
    
    # Setup ArUco detector using OpenCV's ArucoDetector
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    aruco_params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
    
    marker_length = 0.04  # 40mm markers (0.04 m)
    base_marker_id = 8    # Reference marker ID
    object_marker_id = 3  # Object marker ID
    
    # Known transform for the base marker in the manipulator frame.
    # (Note: measured from the marker's center.)
    T_base_marker = np.array([
        [1, 0, 0, -0.09],
        [0, 1, 0,  0.0],
        [0, 0, 1,  0.0],
        [0, 0, 0,  1.0]
    ], dtype=np.float32)
    
    # For fusing the measurements, we initialize a 6D Kalman Filter.
    # Initial state (position set to [0, 0, 1] m, zero velocity)
    initial_state = np.array([[0.0], [0.0], [1.0],
                              [0.0], [0.0], [0.0]])
    dt_init = 1.0 / 30.0  # initial time interval based on 30 FPS
    kf = KalmanFilter6D(dt_init, initial_state)
    
    # We'll use a counter to accumulate valid fusion updates before returning.
    valid_updates = 0
    required_valid_updates = 10  # number of valid fusion updates required
    
    fused_position = None
    T_base_camera = None  # Transformation from camera frame to manipulator base frame

    # Align frames to color stream.
    align_to = rs.stream.color
    align = rs.align(align_to)
    
    prev_time = time.time()
    
    try:
        while True:
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue
            
            color_image = np.asanyarray(color_frame.get_data())
            gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = detector.detectMarkers(gray)
            if ids is not None:
                cv2.aruco.drawDetectedMarkers(color_image, corners, ids)
            
            # -----------------------------
            # Compute T_base_camera using the base marker (ID 8)
            # -----------------------------
            if ids is not None:
                for i, marker_id in enumerate(ids.flatten()):
                    if marker_id == base_marker_id:
                        rvec_base, tvec_base, _ = cv2.aruco.estimatePoseSingleMarkers(
                            [corners[i]], marker_length, camera_matrix, dist_coeffs
                        )
                        rvec_base = rvec_base[0][0]
                        tvec_base = tvec_base[0][0]
                        T_camera_marker = build_transform(rvec_base, tvec_base)
                        T_marker_camera = invert_transform(T_camera_marker)
                        T_base_camera = T_base_marker @ T_marker_camera
                        cv2.drawFrameAxes(color_image, camera_matrix, dist_coeffs,
                                          rvec_base, tvec_base, marker_length * 0.5)
                        break
            
            # -----------------------------
            # Detect object marker (ID 3), compute measurements, and fuse them.
            # -----------------------------
            measurement_pose = None   # From pose estimation
            measurement_depth = None  # From direct depth measurement (back-projection)
            
            if T_base_camera is not None and ids is not None:
                for i, marker_id in enumerate(ids.flatten()):
                    if marker_id == object_marker_id:
                        # Obtain pose estimation for object marker
                        rvec_obj, tvec_obj, _ = cv2.aruco.estimatePoseSingleMarkers(
                            [corners[i]], marker_length, camera_matrix, dist_coeffs
                        )
                        rvec_obj = rvec_obj[0][0]
                        tvec_obj = tvec_obj[0][0]
                        T_camera_obj = build_transform(rvec_obj, tvec_obj)
                        T_base_obj = T_base_camera @ T_camera_obj
                        measurement_pose = T_base_obj[:3, 3]  # 3D position from pose

                        # Also, compute a direct depth measurement.
                        marker_corners = corners[i].reshape((4, 2))
                        center_x = int(np.mean(marker_corners[:, 0]))
                        center_y = int(np.mean(marker_corners[:, 1]))
                        depth_value = depth_frame.get_distance(center_x, center_y)
                        if depth_value > 0:
                            # Back-project pixel to a 3D point in the camera frame
                            x_cam = (center_x - intrinsics.ppx) * depth_value / intrinsics.fx
                            y_cam = (center_y - intrinsics.ppy) * depth_value / intrinsics.fy
                            point_cam = np.array([x_cam, y_cam, depth_value, 1.0], dtype=np.float32)
                            # Transform to manipulator base frame:
                            point_base = T_base_camera @ point_cam
                            measurement_depth = point_base[:3]
                        
                        # Draw marker axes for visualization.
                        cv2.drawFrameAxes(color_image, camera_matrix, dist_coeffs,
                                          rvec_obj, tvec_obj, marker_length * 0.5)
                        # Annotate the measurements.
                        cv2.putText(color_image,
                                    f"Pose: {measurement_pose[0]:.3f}, {measurement_pose[1]:.3f}, {measurement_pose[2]:.3f}",
                                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        if measurement_depth is not None:
                            cv2.putText(color_image,
                                        f"Depth: {measurement_depth[0]:.3f}, {measurement_depth[1]:.3f}, {measurement_depth[2]:.3f}",
                                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                        break

            # Compute elapsed time and update the Kalman filter.
            current_time = time.time()
            dt = current_time - prev_time
            prev_time = current_time
            kf.predict(dt)
            
            # Update KF with available measurements.
            if measurement_pose is not None:
                kf.update(measurement_pose, kf.R_pose)
            if measurement_depth is not None:
                kf.update(measurement_depth, kf.R_depth)
            
            # Extract the fused position from the filter state.
            fused_position = kf.x[:3, 0]
            cv2.putText(color_image,
                        f"Fused Pos: {fused_position[0]:.3f}, {fused_position[1]:.3f}, {fused_position[2]:.3f}",
                        (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.imshow("Fused Pose Detection", color_image)
            key = cv2.waitKey(1)
            
            # Count valid updates (when the object marker is observed).
            if measurement_pose is not None or measurement_depth is not None:
                valid_updates += 1
            
            # Break after a certain number of valid updates or upon key press.
            if valid_updates >= required_valid_updates or (key & 0xFF == ord('q')):
                print("Fused object position in base frame:", fused_position)
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
    
    # Return the fused position
    return fused_position

#######################################
# Motion Functions: Manipulator Movement
#######################################
def move_manipulator_above_marker(base, pos_base):
    """
    Given an object marker position (pos_base) in the base frame,
    move the manipulator to 0.15 m above that location.
    """
    # Round and convert coordinates to floats.
    x = round(float(pos_base[0]), 3)
    y = round(float(pos_base[1]), 3)
    z = round(float(pos_base[2]), 3)
    
    # Target position: same X, Y; Z increased by 0.15 m.
    target_position = [x, y, z + 0.15]
    
    # Fixed orientation (in degrees); adjust if necessary.
    theta_x = 90.0
    theta_y = 0.0
    theta_z = 90.0
    waypoint_tuple = (target_position[0], target_position[1], target_position[2],
                      0.0, theta_x, theta_y, theta_z)
    
    # Create the waypoint list message.
    waypoints = Base_pb2.WaypointList()
    waypoints.duration = 0.0
    waypoints.use_optimal_blending = True
    waypoint = waypoints.waypoints.add()
    waypoint.name = "move_above_marker"
    waypoint.cartesian_waypoint.CopyFrom(populateCartesianCoordinate(waypoint_tuple))
    
    # Validate trajectory.
    result = base.ValidateWaypointList(waypoints)
    if len(result.trajectory_error_report.trajectory_error_elements) != 0:
        print("Trajectory validation failed:")
        for error in result.trajectory_error_report.trajectory_error_elements:
            print("Error:", error)
        return False
    
    # Subscribe to notifications.
    e = threading.Event()
    notification_handle = base.OnNotificationActionTopic(
        check_for_end_or_abort(e),
        Base_pb2.NotificationOptions()
    )
    
    print("Moving manipulator above marker to:", target_position)
    base.ExecuteWaypointTrajectory(waypoints)
    time.sleep(0.5)
    finished = e.wait(TIMEOUT_DURATION)
    base.Unsubscribe(notification_handle)
    
    if finished:
        print("Manipulator reached the target above the marker.")
        open_gripper(base)
        time.sleep(1.0)  # Wait for the gripper to open
    else:
        print("Manipulator movement timed out.")
    return finished

def example_move_to_home_position(base):
    # Make sure the arm is in Single Level Servoing mode.
    base_servo_mode = Base_pb2.ServoingModeInformation()
    base_servo_mode.servoing_mode = Base_pb2.SINGLE_LEVEL_SERVOING
    base.SetServoingMode(base_servo_mode)
    
    # Move arm to ready (home) position.
    print("Moving the arm to a safe position")
    action_type = Base_pb2.RequestedActionType()
    action_type.action_type = Base_pb2.REACH_JOINT_ANGLES
    action_list = base.ReadAllActions(action_type)
    action_handle = None
    for action in action_list.action_list:
        if action.name == "Home":
            action_handle = action.handle

    if action_handle is None:
        print("Can't reach safe position. Exiting")
        return False

    e = threading.Event()
    notification_handle = base.OnNotificationActionTopic(
        check_for_end_or_abort(e),
        Base_pb2.NotificationOptions()
    )

    base.ExecuteActionFromReference(action_handle)
    finished = e.wait(TIMEOUT_DURATION)
    base.Unsubscribe(notification_handle)

    if finished:
        print("Safe position reached")
    else:
        print("Timeout on action notification wait")
    return finished

def move_manipulator_down_and_grip(base, pos_base):
    """
    Given an object marker position (pos_base) in the base frame,
    move the manipulator downward to the level of the marker,
    then close the gripper.
    """
    # Round and convert coordinates to floats.
    x = round(float(pos_base[0]), 3)
    y = round(float(pos_base[1]), 3)
    z = round(float(pos_base[2]), 3)
    
    # Target position: maintain x and y, but set z to the marker level.
    target_position = [x, y, z]
    
    # Fixed orientation (in degrees); adjust as needed.
    theta_x = 90.0
    theta_y = 0.0
    theta_z = 90.0
    waypoint_tuple = (target_position[0], target_position[1], target_position[2],
                      0.0, theta_x, theta_y, theta_z)
    
    # Create the waypoint list message.
    waypoints = Base_pb2.WaypointList()
    waypoints.duration = 0.0
    waypoints.use_optimal_blending = True
    waypoint = waypoints.waypoints.add()
    waypoint.name = "move_down_to_marker"
    waypoint.cartesian_waypoint.CopyFrom(populateCartesianCoordinate(waypoint_tuple))
    
    # Validate the trajectory.
    result = base.ValidateWaypointList(waypoints)
    if len(result.trajectory_error_report.trajectory_error_elements) != 0:
        print("Trajectory validation for downward motion failed:")
        for error in result.trajectory_error_report.trajectory_error_elements:
            print("Error:", error)
        return False
    
    # Subscribe to notifications.
    e = threading.Event()
    notification_handle = base.OnNotificationActionTopic(
        check_for_end_or_abort(e),
        Base_pb2.NotificationOptions()
    )
    
    print("Moving manipulator downward to marker level at:", target_position)
    base.ExecuteWaypointTrajectory(waypoints)
    time.sleep(0.5)
    finished = e.wait(TIMEOUT_DURATION)
    base.Unsubscribe(notification_handle)
    
    if finished:
        print("Manipulator reached the marker level.")
        close_gripper(base)
        time.sleep(1.0)  # Wait for the gripper to close
    else:
        print("Manipulator downward motion timed out.")
    
    return finished

def open_gripper(base):
    """
    Opens the gripper by sending a position command (0.0 for fully open).
    """
    gripper_command = Base_pb2.GripperCommand()
    finger = gripper_command.gripper.finger.add()
    gripper_command.mode = Base_pb2.GRIPPER_POSITION
    finger.finger_identifier = 1
    finger.value = 0.0  # Fully open
    base.SendGripperCommand(gripper_command)
    print("Gripper opened.")

def close_gripper(base):
    """
    Closes the gripper by sending a position command (1.0 for fully closed).
    """
    gripper_command = Base_pb2.GripperCommand()
    finger = gripper_command.gripper.finger.add()
    gripper_command.mode = Base_pb2.GRIPPER_POSITION
    finger.finger_identifier = 1
    finger.value = 1.0  # Fully closed  
    base.SendGripperCommand(gripper_command)
    print("Gripper closed.")

#######################################
# Main Function 
#######################################
def main():
    # Import helper utilities if needed.
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    import utilities

    # Parse connection arguments.
    args = utilities.parseConnectionArguments()
    
    # Create connection to the device.
    with utilities.DeviceConnection.createTcpConnection(args) as router:
        # Create required services.
        base = BaseClient(router)
        base_cyclic = BaseCyclicClient(router)
        
        success = True

        # Move to a safe home position.
        success &= example_move_to_home_position(base)
        
        # Run camera detection to obtain the fused object position in the base frame.
        fused_position = run_camera_detection()
        if fused_position is None:
            print("Object marker not detected. Exiting.")
            return 1
        
        # Command the manipulator to move 0.15 m above the marker.
        success &= move_manipulator_above_marker(base, fused_position)
        
        # Command the manipulator to move downward to the marker level and grip it.
        success &= move_manipulator_down_and_grip(base, fused_position)

        return 0 if success else 1

if __name__ == "__main__":
    exit(main())
