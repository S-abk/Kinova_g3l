########################################################################
# Kinova Gen3 Lite Manipulator with RealSense 435if Camera
# Pick and Place using ArUco Markers
# This script demonstrates the use of Kinova Gen3 lite manipulator with a RealSense 435if camera
# to pick and place an object using ArUco markers for localization.
# The script includes functions to move the manipulator to a safe position, detect ArUco markers,
# move the manipulator above the detected marker, grip the object, and place it at a new location.
# The script uses the Kinova API for robot control and OpenCV for image processing and ArUco detection.
######################################################################


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

def run_camera_detection_for_marker(target_marker_id):
    """
    Runs the RealSense pipeline and ArUco detection until the specified target marker 
    (target_marker_id) is detected. It uses a reference marker (ID 8) to compute the camera-to-base 
    transform and returns the target marker's position in the manipulator base frame.
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

    marker_length = 0.04  # Marker size in meters
    base_marker_id = 8    # Reference marker ID used for establishing the base frame

    # Known transform for the base marker in the manipulator frame.
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

            # Use the reference marker (base_marker_id) to compute T_base_camera.
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

            # Look for the target marker.
            if T_base_camera is not None and ids is not None:
                for i, marker_id in enumerate(ids.flatten()):
                    if marker_id == target_marker_id:
                        rvec_target, tvec_target, _ = cv2.aruco.estimatePoseSingleMarkers(
                            [corners[i]], marker_length, camera_matrix, dist_coeffs
                        )
                        rvec_target = rvec_target[0][0]
                        tvec_target = tvec_target[0][0]
                        T_camera_target = build_transform(rvec_target, tvec_target)
                        T_base_target = T_base_camera @ T_camera_target
                        pos_base = T_base_target[:3, 3]
                        cv2.drawFrameAxes(color_image, camera_matrix, dist_coeffs,
                                          rvec_target, tvec_target, marker_length * 0.5)
                        cv2.putText(color_image,
                                    f"Marker {target_marker_id}: {pos_base[0]:.3f}, {pos_base[1]:.3f}, {pos_base[2]:.3f}",
                                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        break

            cv2.imshow("ArUco Detection", color_image)
            key = cv2.waitKey(1)
            if pos_base is not None or (key & 0xFF == ord('q')):
                if pos_base is not None:
                    print(f"Detected marker {target_marker_id} position in base frame:", pos_base)
                break
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

    return pos_base


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
# Move to Home Position
#######################################
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

# Global fixed orientation (in degrees) used for all motions.
FIXED_ORIENTATION = (90.0, 0.0, 90.0)

#######################################
# Updated: Move Manipulator Above Marker with Clearance Parameter
#######################################
def move_manipulator_above_marker(base, pos_base, clearance=0.15):
    """
    Moves the manipulator to a position above the provided marker position.
    The z-axis clearance is set by the clearance parameter (default 0.15 m).
    """
    x = round(float(pos_base[0]), 3)
    y = round(float(pos_base[1]), 3)
    z = round(float(pos_base[2]), 3)
    
    # Target position: maintain same x and y (with offset) and z increased by clearance.
    target_position = [x - 0.04, y, z + clearance]
    
    theta_x, theta_y, theta_z = FIXED_ORIENTATION
    waypoint_tuple = (target_position[0], target_position[1], target_position[2],
                      0.0, theta_x, theta_y, theta_z)
    
    waypoints = Base_pb2.WaypointList()
    waypoints.duration = 0.0
    waypoints.use_optimal_blending = True
    wp = waypoints.waypoints.add()
    wp.name = "move_above_marker"
    wp.cartesian_waypoint.CopyFrom(populateCartesianCoordinate(waypoint_tuple))
    
    result = base.ValidateWaypointList(waypoints)
    if len(result.trajectory_error_report.trajectory_error_elements) != 0:
        print("Trajectory validation failed:")
        for error in result.trajectory_error_report.trajectory_error_elements:
            print("Error:", error)
        return False
    
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

#######################################
# Function for Downward Motion to Release (Placement)
#######################################
def move_manipulator_down_and_release(base, pos_base):
    """
    Moves the manipulator downward to the level of the provided marker and then opens the gripper to release the object.
    """
    x = round(float(pos_base[0]), 3)
    y = round(float(pos_base[1]), 3)
    z = round(float(pos_base[2]), 3)
    
    # Target position: maintain same x and y (with offset) and z equal to marker's level.
    target_position = [x - 0.04, y, z + 0.2]
    
    theta_x, theta_y, theta_z = FIXED_ORIENTATION
    waypoint_tuple = (target_position[0], target_position[1], target_position[2],
                      0.0, theta_x, theta_y, theta_z)
    
    waypoints = Base_pb2.WaypointList()
    waypoints.duration = 0.0
    waypoints.use_optimal_blending = True
    wp = waypoints.waypoints.add()
    wp.name = "move_down_to_place_marker"
    wp.cartesian_waypoint.CopyFrom(populateCartesianCoordinate(waypoint_tuple))
    
    result = base.ValidateWaypointList(waypoints)
    if len(result.trajectory_error_report.trajectory_error_elements) != 0:
        print("Trajectory validation for downward placement failed:")
        for error in result.trajectory_error_report.trajectory_error_elements:
            print("Error:", error)
        return False
    
    e = threading.Event()
    notification_handle = base.OnNotificationActionTopic(
        check_for_end_or_abort(e),
        Base_pb2.NotificationOptions()
    )
    
    print("Moving manipulator downward to place marker at:", target_position)
    base.ExecuteWaypointTrajectory(waypoints)
    time.sleep(0.5)
    finished = e.wait(TIMEOUT_DURATION)
    base.Unsubscribe(notification_handle)
    
    if finished:
        print("Manipulator reached the place marker level.")
        # Release the object by opening the gripper.
        open_gripper(base)
    else:
        print("Manipulator downward placement motion timed out.")
    
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
    target_position = [x - 0.04, y, z]  # using the same x-offset as before
    
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

#######################################
# Move Straight Up Function
#######################################
def move_straight_up(base, pos_base, delta_z=0.2):
    """
    Moves the manipulator vertically upward from its current gripping position.
    Assumes the gripping position is at the object's marker level, which is given by pos_base.
    The target position is calculated by adding delta_z (default 0.2 m) to the marker's z coordinate.
    """
    # Extract the marker's position and apply the same x-offset used elsewhere.
    x = round(float(pos_base[0]), 3)
    y = round(float(pos_base[1]), 3)
    z = round(float(pos_base[2]), 3)
    
    # The gripping position is assumed to be at [x - 0.04, y, z]. Now, move straight up.
    target_position = [x - 0.04, y, z + delta_z]
    
    theta_x, theta_y, theta_z = FIXED_ORIENTATION
    waypoint_tuple = (target_position[0], target_position[1], target_position[2],
                      0.0, theta_x, theta_y, theta_z)
    
    waypoints = Base_pb2.WaypointList()
    waypoints.duration = 0.0
    waypoints.use_optimal_blending = True
    wp = waypoints.waypoints.add()
    wp.name = "move_straight_up"
    wp.cartesian_waypoint.CopyFrom(populateCartesianCoordinate(waypoint_tuple))
    
    # Validate the trajectory.
    result = base.ValidateWaypointList(waypoints)
    if len(result.trajectory_error_report.trajectory_error_elements) != 0:
        print("Trajectory validation for moving straight up failed:")
        for error in result.trajectory_error_report.trajectory_error_elements:
            print("Error:", error)
        return False
    
    # Execute the trajectory.
    e = threading.Event()
    notification_handle = base.OnNotificationActionTopic(
        check_for_end_or_abort(e),
        Base_pb2.NotificationOptions()
    )
    
    print("Moving manipulator straight up by", delta_z, "meters to:", target_position)
    base.ExecuteWaypointTrajectory(waypoints)
    time.sleep(0.5)
    finished = e.wait(TIMEOUT_DURATION)
    base.Unsubscribe(notification_handle)
    
    if finished:
        print("Manipulator moved straight up successfully.")
    else:
        print("Manipulator straight upward movement timed out.")
    return finished


#######################################
# Main Routine
#######################################
def main():
    # Import the utilities helper module
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    import utilities

    # Parse arguments and establish connection
    args = utilities.parseConnectionArguments()
    with utilities.DeviceConnection.createTcpConnection(args) as router:
        base = BaseClient(router)
        base_cyclic = BaseCyclicClient(router)
        
        # Move to a safe home position.
        success = example_move_to_home_position(base)
        
        # --- Picking Sequence ---
        # Run camera detection to obtain the object marker's base-frame position.
        pos_obj = run_camera_detection_for_marker(3)
        # Marker ID 3 is used for the object.
        if pos_obj is None:
            print("Object marker not detected. Exiting.")
            return 1
        
        # Move to 0.15 m above the object marker.
        success &= move_manipulator_above_marker(base, pos_obj, clearance=0.15)
        # Lower to the object and grip.
        success &= move_manipulator_down_and_grip(base, pos_obj)

        # --- Move Straight Up to Avoid Collisions ---
        success &= move_straight_up(base, pos_obj, delta_z=0.2)
        
        # --- Placement Sequence ---
        # Let the user (or external logic) select the placement marker id.
        target_marker_id = int(input("Enter the placement marker id (e.g., 0, 1, or 2): "))
        pos_target = run_camera_detection_for_marker(target_marker_id)
        if pos_target is None:
            print("Placement marker not detected. Exiting.")
            return 1
        
        # For placement, use a higher clearance (0.30 m) to account for object height.
        success &= move_manipulator_above_marker(base, pos_target, clearance=0.30)
        # Lower to the placement marker and release the object.
        success &= move_manipulator_down_and_release(base, pos_target)
        
        # Optionally, add a retract or safe position move here.
        return 0 if success else 1

if __name__ == "__main__":
    exit(main())
