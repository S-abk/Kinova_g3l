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

def run_camera_detection():
    """
    Runs the RealSense pipeline and ArUco detection until the object marker (ID 3)
    is detected. Computes and returns the object's pose in the manipulator base frame.
    """
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    profile = pipeline.start(config)
    
    # Get camera intrinsics.
    color_stream = profile.get_stream(rs.stream.color)
    color_profile = color_stream.as_video_stream_profile()
    intrinsics = color_profile.get_intrinsics()
    camera_matrix = np.array([[intrinsics.fx, 0, intrinsics.ppx],
                               [0, intrinsics.fy, intrinsics.ppy],
                               [0, 0, 1]])
    dist_coeffs = np.array(intrinsics.coeffs)
    print("Camera matrix:\n", camera_matrix)
    print("Distortion coefficients:\n", dist_coeffs)
    
    # Setup ArUco detection.
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    aruco_params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
    
    marker_length = 0.04  # 40mm markers (0.04 m)
    base_marker_id = 8    # Reference marker ID
    object_marker_id = 3  # Object marker ID
    
    # Known transform for the base marker in the manipulator frame.
    # Its origin is offset -0.09 m in the X direction (with no rotation).
    T_base_marker = np.array([
        [1, 0, 0, -0.09],
        [0, 1, 0,  0.0],
        [0, 0, 1,  0.0],
        [0, 0, 0,  1.0]
    ], dtype=np.float32)
    
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
                        # Once detected, break out.
                        break
            
            cv2.imshow("ArUco Detection", color_image)
            key = cv2.waitKey(1)
            if pos_base is not None or (key & 0xFF == ord('q')):
                if pos_base is not None:
                    print("Detected object marker position in base frame:", pos_base)
                break
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
    
    return pos_base

#######################################
# Motion Function: Move Above Marker
#######################################
def move_manipulator_above_marker(base, pos_base):
    """
    Given an object marker position (pos_base) in the base frame,
    move the manipulator to 0.15 m above that location.
    """
    # Round and convert coordinates to native Python floats.
    x = round(float(pos_base[0]), 3)
    y = round(float(pos_base[1]), 3)
    z = round(float(pos_base[2]), 3)
    
    # Target position: same X, Y; Z increased by 0.15 m.
    target_position = [x - 0.05, y, z + 0.15]
    
    # Fixed orientation (in degrees); adjust if necessary.
    theta_x = 90.0
    theta_y = 0.0
    theta_z = 90.0
    waypoint_tuple = (target_position[0], target_position[1], target_position[2],
                      0.0, theta_x, theta_y, theta_z)
    
    # Create the waypoint list message.
    waypoints = Base_pb2.WaypointList()
    # Set a duration that meets the minimum requirement (e.g., 2.5 seconds).
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
        time.sleep(1.0)  # Wait for the gripper to close
    else:
        print("Manipulator movement timed out.")
    return finished


def example_move_to_home_position(base):
    # Make sure the arm is in Single Level Servoing mode
    base_servo_mode = Base_pb2.ServoingModeInformation()
    base_servo_mode.servoing_mode = Base_pb2.SINGLE_LEVEL_SERVOING
    base.SetServoingMode(base_servo_mode)
    
    # Move arm to ready position
    print("Moving the arm to a safe position")
    action_type = Base_pb2.RequestedActionType()
    action_type.action_type = Base_pb2.REACH_JOINT_ANGLES
    action_list = base.ReadAllActions(action_type)
    action_handle = None
    for action in action_list.action_list:
        if action.name == "Home":
            action_handle = action.handle

    if action_handle == None:
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
    move the manipulator downward (in the z axis) to the level of the marker,
    and then close the gripper.
    """
    # Round and convert coordinates to floats.
    x = round(float(pos_base[0]), 3)
    y = round(float(pos_base[1]), 3)
    z = round(float(pos_base[2]), 3)
    
    # Compute target position: maintain same x and y (with offset) but set z to the marker's level.
    target_position = [x - 0.05, y, z]  # using the same x-offset as before
    
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
        # Once at the marker, close the gripper.
        close_gripper(base)
        time.sleep(1.0)  # Wait for the gripper to close
    else:
        print("Manipulator downward motion timed out.")
    
    return finished

def open_gripper(base):
    """
    Opens the gripper by sending a position command (1.0 for fully open).
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
    Closes the gripper by sending a position command (0.0 for fully closed).
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
    # Import the utilities helper module
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    import utilities

    # Parse arguments
    args = utilities.parseConnectionArguments()
    
    # Create connection to the device and get the router
    with utilities.DeviceConnection.createTcpConnection(args) as router:
        # Create required services
        base = BaseClient(router)
        base_cyclic = BaseCyclicClient(router)
        
        # Example core
        success = True

        # Move to a safe home position (existing function)
        success &= example_move_to_home_position(base)
        
        # Run camera detection to obtain the object marker's base-frame position.
        pos_base = run_camera_detection()
        if pos_base is None:
            print("Object marker not detected. Exiting.")
            return 1
        
        # Command the manipulator to move to 0.15 m above the marker.
        success &= move_manipulator_above_marker(base, pos_base)
        
        # Command the manipulator to move down to the marker level and grip it.
        success &= move_manipulator_down_and_grip(base, pos_base)



        return 0 if success else 1

if __name__ == "__main__":
    exit(main())
