import pyrealsense2 as rs
import cv2
import numpy as np
import sys
import os
import time
import threading

# -----------------------------
# Manipulator API Imports & Constants
# -----------------------------
from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import BaseCyclicClient
from kortex_api.autogen.messages import Base_pb2, BaseCyclic_pb2, Common_pb2

TIMEOUT_DURATION = 30

def check_for_end_or_abort(e):
    """Closure for monitoring action completion."""
    def check(notification, e=e):
        print("EVENT: " + Base_pb2.ActionEvent.Name(notification.action_event))
        if (notification.action_event == Base_pb2.ACTION_END or 
            notification.action_event == Base_pb2.ACTION_ABORT):
            e.set()
    return check

def populateCartesianCoordinate(waypointInformation):
    """
    Populates a CartesianWaypoint message.
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

def move_manipulator_above_marker(base, pos_base):
    """
    Given an object marker position (pos_base) in the base frame,
    move the manipulator to 0.15 m above that location.
    """
    # Convert numpy floats to native Python floats.
    x = round(float(pos_base[0]), 3)
    y = round(float(pos_base[1]), 3)
    z = round(float(pos_base[2]), 3)
    
    # Calculate target position (0.15 m above means increasing Z by 0.15)
    target_position = [x, y, z + 0.15]
    
    # Use a fixed orientation (in degrees). 
    # If the current orientation is causing issues, try simpler values (e.g., 0, 90, 0)
    theta_x = 10.772   # try adjusting these if needed
    theta_y = 177.866
    theta_z = 82.777
    waypoint_tuple = (target_position[0], target_position[1], target_position[2],
                        0.0, theta_x, theta_y, theta_z)
    
    # Create a waypoint list message for the trajectory.
    waypoints = Base_pb2.WaypointList()
    # Specify a non-zero duration (in seconds) for a smooth move.
    waypoints.duration = 0.0  
    waypoints.use_optimal_blending = True
    waypoint = waypoints.waypoints.add()
    waypoint.name = "move_above_marker"
    waypoint.cartesian_waypoint.CopyFrom(populateCartesianCoordinate(waypoint_tuple))
    
    # Validate the trajectory.
    result = base.ValidateWaypointList(waypoints)
    if len(result.trajectory_error_report.trajectory_error_elements) != 0:
        print("Trajectory validation failed:")
        for error in result.trajectory_error_report.trajectory_error_elements:
            print("Error:", error)
        return False
    
    # Subscribe to notifications to monitor the action.
    e = threading.Event()
    notification_handle = base.OnNotificationActionTopic(
        check_for_end_or_abort(e),
        Base_pb2.NotificationOptions()
    )
    
    print("Moving manipulator above marker to:", target_position)
    base.ExecuteWaypointTrajectory(waypoints)
    # Allow some time after issuing the command
    time.sleep(0.5)
    finished = e.wait(TIMEOUT_DURATION)
    base.Unsubscribe(notification_handle)
    
    if finished:
        print("Manipulator reached the target above the marker.")
    else:
        print("Manipulator movement timed out.")
    return finished

# -----------------------------
# RealSense & ArUco Detection
# -----------------------------
def run_camera_detection():
    """
    Runs the RealSense pipeline and ArUco detection until the object marker (ID 3)
    is detected. Computes the object's pose in the manipulator base frame.
    """
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    profile = pipeline.start(config)
    
    # Get camera intrinsics for pose estimation.
    color_stream = profile.get_stream(rs.stream.color)
    color_profile = color_stream.as_video_stream_profile()
    intrinsics = color_profile.get_intrinsics()
    camera_matrix = np.array([[intrinsics.fx, 0, intrinsics.ppx],
                               [0, intrinsics.fy, intrinsics.ppy],
                               [0, 0, 1]])
    dist_coeffs = np.array(intrinsics.coeffs)
    print("Camera matrix:\n", camera_matrix)
    print("Distortion coefficients:\n", dist_coeffs)
    
    # Set up ArUco detection.
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    aruco_params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
    
    marker_length = 0.04  # 40mm markers (0.04 m)
    base_marker_id = 8    # Known base marker
    object_marker_id = 3  # Object marker to pick
    
    # Known transform for the base marker in the manipulator frame:
    # It is translated -0.07 m in X with no rotation.
    T_base_marker = np.array([
        [1, 0, 0, -0.07],
        [0, 1, 0,  0.0 ],
        [0, 0, 1,  0.0 ],
        [0, 0, 0,  1.0 ]
    ], dtype=np.float32)
    
    def build_transform(rvec, tvec):
        R, _ = cv2.Rodrigues(rvec)
        T = np.eye(4, dtype=np.float32)
        T[:3, :3] = R
        T[:3, 3] = tvec.reshape(3)
        return T

    def invert_transform(T):
        R = T[:3, :3]
        t = T[:3, 3]
        R_inv = R.T
        t_inv = -R_inv @ t
        T_inv = np.eye(4, dtype=np.float32)
        T_inv[:3, :3] = R_inv
        T_inv[:3, 3] = t_inv
        return T_inv

    pos_base = None
    T_base_camera = None

    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue
            color_image = np.asanyarray(color_frame.get_data())
            gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = detector.detectMarkers(gray)
            if ids is not None:
                cv2.aruco.drawDetectedMarkers(color_image, corners, ids)
            
            # Find the base marker (ID 8) to compute T_base_camera.
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
            
            # Find the object marker (ID 3) and compute its pose in the base frame.
            if T_base_camera is not None and ids is not None:
                for i, marker_id in enumerate(ids.flatten()):
                    if marker_id == object_marker_id:
                        rvec_obj, tvec_obj, _ = cv2.aruco.estimatePoseSingleMarkers(
                            [corners[i]], marker_length, camera_matrix, dist_coeffs
                        )
                        rvec_obj = rvec_obj[0][0]
                        tvec_obj = tvec_obj[0][0]
                        T_camera_obj = build_transform(rvec_obj, tvec_obj)
                        T_base_obj = T_base_camera @ T_camera_obj
                        pos_base = T_base_obj[:3, 3]
                        cv2.drawFrameAxes(color_image, camera_matrix, dist_coeffs,
                                          rvec_obj, tvec_obj, marker_length * 0.5)
                        cv2.putText(color_image,
                                    f"Obj Pos: {pos_base[0]:.3f}, {pos_base[1]:.3f}, {pos_base[2]:.3f}",
                                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        # Once the object marker is detected, exit the loop.
                        break

            cv2.imshow("ArUco Detection", color_image)
            key = cv2.waitKey(1)
            if pos_base is not None:
                print("Detected object marker position in base frame:", pos_base)
                break
            if key & 0xFF == ord('q'):
                break
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
    
    return pos_base

# -----------------------------
# Main Function: Integration
# -----------------------------
def main():
    # Establish connection to the manipulator (using your utilities for connection)
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    import utilities
    args = utilities.parseConnectionArguments()
    with utilities.DeviceConnection.createTcpConnection(args) as router:
        base = BaseClient(router)
        base_cyclic = BaseCyclicClient(router)
        
        # Set the robot to Single Level Servoing mode.
        base_servo_mode = Base_pb2.ServoingModeInformation()
        base_servo_mode.servoing_mode = Base_pb2.SINGLE_LEVEL_SERVOING
        base.SetServoingMode(base_servo_mode)
        
        # Run the camera detection to obtain the object marker's position.
        pos_base = run_camera_detection()
        if pos_base is None:
            print("Object marker not detected. Exiting.")
            return 1
        
        # Command the manipulator to move to 0.15 m above the marker location.
        if not move_manipulator_above_marker(base, pos_base):
            print("Failed to move manipulator above the marker.")
            return 1
        
        # Proceed with further operations (e.g., lowering to pick)
        print("Manipulator is above the marker. Proceeding with subsequent operations...")
        # Insert additional motion or pick/place routines here.
        
        return 0

if __name__ == "__main__":
    exit(main())