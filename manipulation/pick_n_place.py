import os
import sys
import time
import threading
import numpy as np
import cv2
import pyrealsense2 as rs
import math

# Manipulator API Imports
from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import BaseCyclicClient
from kortex_api.autogen.messages import Base_pb2, BaseCyclic_pb2, Common_pb2

# Task Parameters
TIMEOUT_DURATION = 30
PICK_MARKER_ID = 3
PLACE_MARKER_ID = 0
BASE_MARKER_ID = 8
MARKER_SIZE = 0.04 # meters (e.g., 40mm)

# --- Offsets relative to marker origins ---
# Approach Point above Pick Marker (ID 3)
PICK_APPROACH_POS_OFFSET = {'x': 0.0, 'y': 0.15, 'z': 0.05}
# Grasp Point relative to Pick Marker (ID 3) - slightly above marker plane
PICK_GRASP_POS_OFFSET    = {'x': 0.0, 'y': 0.0, 'z': 0.05}
# Approach Point above Place Marker (ID 0)
PLACE_APPROACH_POS_OFFSET= {'x': 0.0, 'y': 0.2, 'z': -0.03}
# Release Point relative to Place Marker (ID 0) - 50mm below marker origin
PLACE_RELEASE_POS_OFFSET = {'x': 0.0, 'y': 0.13, 'z': -0.03}

# --- Target Orientation in Base Frame ---
# (Desired final ThetaX, ThetaY, ThetaZ for the end-effector in the robot base frame)
TARGET_ORIENTATION_BASE = {'theta_x': 90.0, 'theta_y': 0.0, 'theta_z': 90.0} # Pointing down

# Known transform for the base marker (ID 8) in the manipulator frame.
# Adjust this if your base marker is placed differently.
T_base_marker = np.array([
    [1, 0, 0, -0.09],
    [0, 1, 0,  0.0],
    [0, 0, 1,  0.0],
    [0, 0, 0,  1.0]
], dtype=np.float32)

#######################################
# Transformation Helper Functions
#######################################

# Checks if a matrix is a valid rotation matrix.
def isRotationMatrix(R):
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6

# Calculates rotation matrix to euler angles
# The result is XYZ-intrinsic angles, which correspond to ZYX-extrinsic (ThetaX, ThetaY, ThetaZ used by Kortex)
def rotationMatrixToEulerAngles(R):
    assert (isRotationMatrix(R))
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6
    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0
    # Convert radians to degrees
    return np.array([math.degrees(x), math.degrees(y), math.degrees(z)])

# Calculates Rotation Matrix given euler angles. ZYX extrinsic (ThetaX, ThetaY, ThetaZ)
def eulerAnglesToRotationMatrix(theta):
    # Convert degrees to radians
    theta_rad = np.radians(theta)
    R_x = np.array([[1,         0,                  0                   ],
                    [0,         math.cos(theta_rad[0]), -math.sin(theta_rad[0]) ],
                    [0,         math.sin(theta_rad[0]), math.cos(theta_rad[0])  ]
                    ])
    R_y = np.array([[math.cos(theta_rad[1]),    0,      math.sin(theta_rad[1])  ],
                    [0,                     1,      0                   ],
                    [-math.sin(theta_rad[1]),   0,      math.cos(theta_rad[1])  ]
                    ])
    R_z = np.array([[math.cos(theta_rad[2]),    -math.sin(theta_rad[2]),    0],
                    [math.sin(theta_rad[2]),    math.cos(theta_rad[2]),     0],
                    [0,                     0,                      1]
                    ])
    # Apply ZYX extrinsic convention (equivalent to roll, pitch, yaw)
    R = np.dot(R_z, np.dot(R_y, R_x))
    return R

def build_transform(rvec, tvec):
    """Build a 4x4 homogeneous transformation matrix T_camera_marker from rvec and tvec."""
    R, _ = cv2.Rodrigues(rvec) # rvec to rotation matrix
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

def build_transform_from_offset_dict(offset_dict):
    """Builds a 4x4 T_marker_offset matrix from an offset dictionary."""
    T = np.eye(4, dtype=np.float32)
    # Create rotation matrix from Euler angles
    R = eulerAnglesToRotationMatrix([offset_dict['theta_x'], offset_dict['theta_y'], offset_dict['theta_z']])
    T[:3, :3] = R
    # Add translation
    T[0, 3] = offset_dict['x']
    T[1, 3] = offset_dict['y']
    T[2, 3] = offset_dict['z']
    return T

def decompose_transform(T):
    """Decomposes a 4x4 transform T_base_target into components for Kortex API."""
    x = T[0, 3]
    y = T[1, 3]
    z = T[2, 3]
    # Extract rotation matrix part
    R = T[:3, :3]
    # Convert rotation matrix to Euler angles (degrees)
    thetas = rotationMatrixToEulerAngles(R)
    theta_x = thetas[0]
    theta_y = thetas[1]
    theta_z = thetas[2]
    return x, y, z, theta_x, theta_y, theta_z

#######################################
# Robot Action Helper Functions
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
    waypointInformation: (x, y, z, theta_x, theta_y, theta_z, blending_radius)
    """
    waypoint = Base_pb2.CartesianWaypoint()
    waypoint.pose.x = waypointInformation[0]
    waypoint.pose.y = waypointInformation[1]
    waypoint.pose.z = waypointInformation[2]
    waypoint.pose.theta_x = waypointInformation[3]
    waypoint.pose.theta_y = waypointInformation[4]
    waypoint.pose.theta_z = waypointInformation[5]
    waypoint.blending_radius = waypointInformation[6] # Use index 6 for blending
    waypoint.reference_frame = Base_pb2.CARTESIAN_REFERENCE_FRAME_BASE
    return waypoint

def open_gripper(base):
    """Opens the gripper."""
    gripper_command = Base_pb2.GripperCommand()
    finger = gripper_command.gripper.finger.add()
    gripper_command.mode = Base_pb2.GRIPPER_POSITION
    finger.finger_identifier = 1
    finger.value = 0.0  # Fully open
    print("Opening gripper...")
    base.SendGripperCommand(gripper_command)
    time.sleep(1.5) # Allow time for gripper to open

def close_gripper(base):
    """Closes the gripper."""
    gripper_command = Base_pb2.GripperCommand()
    finger = gripper_command.gripper.finger.add()
    gripper_command.mode = Base_pb2.GRIPPER_POSITION
    finger.finger_identifier = 1
    finger.value = 0.7 # Partially or fully closed - adjust as needed
    print("Closing gripper...")
    base.SendGripperCommand(gripper_command)
    time.sleep(1.5) # Allow time for gripper to close

def example_move_to_home_position(base):
    # Make sure the arm is in Single Level Servoing mode
    base_servo_mode = Base_pb2.ServoingModeInformation()
    base_servo_mode.servoing_mode = Base_pb2.SINGLE_LEVEL_SERVOING
    base.SetServoingMode(base_servo_mode)

    print("Moving the arm to Home position...")
    action_type = Base_pb2.RequestedActionType()
    action_type.action_type = Base_pb2.REACH_JOINT_ANGLES
    action_list = base.ReadAllActions(action_type)
    action_handle = None
    for action in action_list.action_list:
        if action.name == "Home":
            action_handle = action.handle
            break

    if action_handle is None:
        print("FATAL: Cannot find 'Home' action.")
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
        print("Home position reached.")
    else:
        print("Timeout waiting for Home position.")
    return finished

#######################################
# Camera & Detection Function
#######################################
def detect_markers(marker_size, required_marker_ids, base_marker_id, T_base_marker_known):
    """
    Detects ArUco markers, calculates camera pose, and returns poses of detected markers.

    Args:
        marker_size: Physical size of the markers in meters.
        required_marker_ids: List of marker IDs necessary for the task.
        base_marker_id: The ID of the marker used for camera localization.
        T_base_marker_known: The known 4x4 transform of the base marker relative to the robot base.

    Returns:
        tuple: (T_base_camera, detected_markers_camera)
               T_base_camera: 4x4 numpy array (or None if base marker not found).
               detected_markers_camera: Dict {id: T_camera_marker} for all detected markers (or None).
    """
    pipeline = None
    detected_markers_camera = {}
    T_base_camera = None
    intrinsics_obtained = False
    camera_matrix = None
    dist_coeffs = None

    try:
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        profile = pipeline.start(config)
        print("RealSense pipeline started.")

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
        intrinsics_obtained = True

        # Setup ArUco detection
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
        aruco_params = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
        print("ArUco detector initialized.")

        start_time = time.time()
        detection_timeout = 10 # seconds to try detecting

        while time.time() - start_time < detection_timeout:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue
            color_image = np.asanyarray(color_frame.get_data())
            gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = detector.detectMarkers(gray)

            detected_markers_camera.clear() # Clear previous detections in this frame
            T_base_camera = None # Reset T_base_camera calculation for this frame

            if ids is not None:
                rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                    corners, marker_size, camera_matrix, dist_coeffs
                )
                cv2.aruco.drawDetectedMarkers(color_image, corners, ids)

                # Store all detected marker poses relative to camera
                for i, marker_id in enumerate(ids.flatten()):
                    rvec = rvecs[i][0]
                    tvec = tvecs[i][0]
                    T_camera_marker = build_transform(rvec, tvec)
                    detected_markers_camera[marker_id] = T_camera_marker
                    cv2.drawFrameAxes(color_image, camera_matrix, dist_coeffs, rvec, tvec, marker_size * 0.75)

                    # Try to calculate T_base_camera if base marker is detected
                    if marker_id == base_marker_id:
                        T_marker_camera = invert_transform(T_camera_marker)
                        T_base_camera = T_base_marker_known @ T_marker_camera
                        # Optional: Draw axes for the calculated camera pose in base frame for debugging
                        # This requires projecting the base frame origin into the camera view

            # Display detection results
            cv2.imshow("ArUco Detection", color_image)
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                 print("Detection quit by user.")
                 return None, None # Indicate failure

            # Check if we have found the base marker and all required markers
            if T_base_camera is not None:
                all_required_found = True
                for req_id in required_marker_ids:
                    if req_id not in detected_markers_camera:
                        all_required_found = False
                        break
                if all_required_found:
                    print(f"Successfully detected base marker ({base_marker_id}) and required markers: {required_marker_ids}")
                    # Draw final T_base_camera pose for a moment
                    if T_base_camera is not None:
                         # Could add drawing logic here if needed
                         pass
                    cv2.waitKey(500) # Display final frame briefly
                    return T_base_camera, detected_markers_camera

            time.sleep(0.1) # Small delay

        print(f"Timeout reached. Could not detect all required markers and base marker within {detection_timeout}s.")
        return None, None # Indicate failure

    except Exception as e:
        print(f"Error during detection: {e}")
        return None, None # Indicate failure
    finally:
        if pipeline:
            pipeline.stop()
            print("RealSense pipeline stopped.")
        cv2.destroyAllWindows()
        print("Detection window closed.")


#######################################
# Motion Execution Functions
#######################################

def calculate_target_pose_in_base(ref_marker_id, position_offset_dict, target_orientation_base_dict, T_base_camera, detected_markers_camera):
    """
    Calculates the target pose in the robot base frame.
    Position is relative to the marker. Orientation is specified in the base frame.
    """
    if T_base_camera is None:
        print(f"Error: Cannot calculate target pose, T_base_camera is None (base marker {BASE_MARKER_ID} likely not detected).")
        return None
    if ref_marker_id not in detected_markers_camera:
        print(f"Error: Cannot calculate target pose, reference marker {ref_marker_id} not detected.")
        return None

    # 1. Calculate the full pose of the reference marker in the base frame
    T_camera_marker_ref = detected_markers_camera[ref_marker_id]
    T_base_marker_ref = T_base_camera @ T_camera_marker_ref

    # 2. Create a purely positional offset transform relative to the marker frame
    #    (Identity rotation, specified translation)
    T_marker_pos_offset = np.eye(4, dtype=np.float32)
    T_marker_pos_offset[0, 3] = position_offset_dict['x']
    T_marker_pos_offset[1, 3] = position_offset_dict['y']
    T_marker_pos_offset[2, 3] = position_offset_dict['z']

    # 3. Calculate the final target position in the base frame
    #    Apply the positional offset in the marker's frame
    T_base_target_pos_only = T_base_marker_ref @ T_marker_pos_offset
    P_base_target = T_base_target_pos_only[:3, 3] # Extract the final [x, y, z] position vector

    # 4. Create the target rotation matrix from the desired base frame orientation
    R_base_target = eulerAnglesToRotationMatrix([
        target_orientation_base_dict['theta_x'],
        target_orientation_base_dict['theta_y'],
        target_orientation_base_dict['theta_z']
    ])

    # 5. Combine the target position and target rotation into the final 4x4 pose matrix
    T_base_target = np.eye(4, dtype=np.float32)
    T_base_target[:3, :3] = R_base_target
    T_base_target[:3, 3] = P_base_target

    # print(f"Calculated T_base_target relative to marker {ref_marker_id} (Pos Only) with Base Orientation:\n{T_base_target}")
    return T_base_target

def move_to_pose_relative_to_marker(base, ref_marker_id, position_offset_dict, target_orientation_base_dict, T_base_camera, detected_markers_camera, blending_radius=0.0):
    """Calculates target pose relative to a marker (position only) with a specified
       base frame orientation, and moves the robot."""
    print(f"\nAttempting move relative to Marker ID: {ref_marker_id}")
    print(f"Position Offset: {position_offset_dict}")
    print(f"Target Base Orientation: {target_orientation_base_dict}")

    # Call the modified calculation function
    T_base_target = calculate_target_pose_in_base(ref_marker_id, position_offset_dict, target_orientation_base_dict, T_base_camera, detected_markers_camera)

    if T_base_target is None:
        print("Move failed: Could not calculate target pose.")
        return False

    try:
        # Decompose and execute the move (this part remains the same)
        x, y, z, theta_x, theta_y, theta_z = decompose_transform(T_base_target)
        print(f"Target Base Coordinates: x={x:.3f}, y={y:.3f}, z={z:.3f}, tx={theta_x:.1f}, ty={theta_y:.1f}, tz={theta_z:.1f}")

        waypoint_info = (x, y, z, theta_x, theta_y, theta_z, blending_radius)

        waypoints = Base_pb2.WaypointList()
        waypoints.duration = 0.0
        waypoints.use_optimal_blending = False
        waypoint = waypoints.waypoints.add()
        waypoint.name = f"move_rel_marker_{ref_marker_id}_base_ori"
        waypoint.cartesian_waypoint.CopyFrom(populateCartesianCoordinate(waypoint_info))

        print("Validating trajectory...")
        result = base.ValidateWaypointList(waypoints)
        if len(result.trajectory_error_report.trajectory_error_elements) != 0:
            print("FATAL: Trajectory validation failed:")
            print(result.trajectory_error_report)
            return False
        print("Trajectory valid.")

        e = threading.Event()
        notification_handle = base.OnNotificationActionTopic(
            check_for_end_or_abort(e),
            Base_pb2.NotificationOptions()
        )

        print("Executing trajectory...")
        base.ExecuteWaypointTrajectory(waypoints)
        finished = e.wait(TIMEOUT_DURATION)
        base.Unsubscribe(notification_handle)

        if finished:
            print("Move successful.")
        else:
            print("Move timed out.")
        return finished

    except Exception as e:
        print(f"Error during move execution: {e}")
        return False


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
        # base_cyclic = BaseCyclicClient(router) # Not used in this demo

        success = True
        T_base_camera = None
        detected_markers = None

        try:
            # 1. Move to Home Position
            success &= example_move_to_home_position(base)
            if not success: raise RuntimeError("Failed to reach home position.")

            # 2. Open Gripper
            open_gripper(base) # Open before detection/approach

            # 3. Detect Markers
            print("\n--- Starting Marker Detection Phase ---")
            required_markers = [PICK_MARKER_ID, PLACE_MARKER_ID] # We need both for the full task
            T_base_camera, detected_markers = detect_markers(MARKER_SIZE, required_markers, BASE_MARKER_ID, T_base_marker)

            if T_base_camera is None or detected_markers is None:
                 raise RuntimeError("Failed to detect required markers. Exiting.")
            print("--- Marker Detection Successful ---")
            print(f"Calculated T_base_camera:\n{T_base_camera}")
            print("Detected Markers (Camera Frame):")
            for marker_id, T_cam_marker in detected_markers.items():
                 print(f" ID {marker_id}: Position {T_cam_marker[:3, 3]}")


            # --- Pick Sequence (Marker 0) ---
            print("\n--- Starting Pick Sequence ---")
            # 4. Approach Pick Marker (Move above)
            success &= move_to_pose_relative_to_marker(base, PICK_MARKER_ID, PICK_APPROACH_POS_OFFSET, TARGET_ORIENTATION_BASE, T_base_camera, detected_markers)
            if not success: raise RuntimeError("Failed pick approach move.")


            # 5. Descend to Pick Marker
            success &= move_to_pose_relative_to_marker(base, PICK_MARKER_ID, PICK_GRASP_POS_OFFSET, TARGET_ORIENTATION_BASE, T_base_camera, detected_markers)
            if not success: raise RuntimeError("Failed pick descend move.")

            # 6. Close Gripper
            close_gripper(base)
            time.sleep(1.0) # Wait briefly after grasp

            # 7. Ascend from Pick Marker
            success &= move_to_pose_relative_to_marker(base, PICK_MARKER_ID, PICK_APPROACH_POS_OFFSET, TARGET_ORIENTATION_BASE, T_base_camera, detected_markers)
            if not success: raise RuntimeError("Failed pick ascend move.")
            print("--- Pick Sequence Complete ---")

            # --- Place Sequence (Marker 1) ---
            print("\n--- Starting Place Sequence ---")
            # 8. Approach Place Marker (Move above)
            success &= move_to_pose_relative_to_marker(base, PLACE_MARKER_ID, PLACE_APPROACH_POS_OFFSET, TARGET_ORIENTATION_BASE, T_base_camera, detected_markers)
            if not success: raise RuntimeError("Failed place approach move.")

            # 9. Descend to Place Marker (Target Z = -50mm relative)
            success &= move_to_pose_relative_to_marker(base, PLACE_MARKER_ID, PLACE_RELEASE_POS_OFFSET, TARGET_ORIENTATION_BASE, T_base_camera, detected_markers)
            if not success: raise RuntimeError("Failed place descend move.")

            # 10. Open Gripper
            open_gripper(base)
            time.sleep(1.0) # Wait briefly after release

            # 11. Ascend from Place Marker
            success &= move_to_pose_relative_to_marker(base, PLACE_MARKER_ID, PLACE_APPROACH_POS_OFFSET, TARGET_ORIENTATION_BASE, T_base_camera, detected_markers)
            if not success: raise RuntimeError("Failed place ascend move.")
            print("--- Place Sequence Complete ---")


            # 12. Move back Home
            print("\n--- Returning to Home ---")
            success &= example_move_to_home_position(base)

        except Exception as e:
            print(f"\n!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print(f"An error occurred: {e}")
            print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            success = False

        finally:
            # Attempt to move home even if an error occurred mid-task
            if 'base' in locals():
                 print("Ensuring arm is home...")
                 example_move_to_home_position(base)

        return 0 if success else 1

if __name__ == "__main__":
    exit(main())