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
FILTER_DURATION = 1.5 # seconds to run filter refinement

# --- Positional Offsets relative to marker origins ---
PICK_APPROACH_POS_OFFSET = {'x': 0.0, 'y': 0.2, 'z': 0.03}
PICK_GRASP_POS_OFFSET    = {'x': 0.0, 'y': 0.0, 'z': 0.03}
PLACE_APPROACH_POS_OFFSET= {'x': 0.0, 'y': 0.26, 'z': -0.01}
PLACE_RELEASE_POS_OFFSET = {'x': 0.0, 'y': 0.16, 'z': -0.01}

# --- Target Orientation in Base Frame ---
TARGET_ORIENTATION_BASE = {'theta_x': 90.0, 'theta_y': 0.0, 'theta_z': 90.0} # Pointing down

# Known transform for the base marker (ID 8)
T_base_marker = np.array([
    [1, 0, 0, -0.09],
    [0, 1, 0,  0.0],
    [0, 0, 1,  0.0],
    [0, 0, 0,  1.0]
], dtype=np.float32)


#######################################
# Kalman Filter Class (from provided script)
#######################################
class KalmanFilter6D:
    def __init__(self, dt, initial_state):
        # State vector: [x, y, z, vx, vy, vz]^T (shape 6,1)
        self.x = initial_state.reshape(6,1) if initial_state is not None else np.zeros((6, 1))
        # Initial state covariance
        self.P = np.eye(6) * 1.0 # Start with relatively high uncertainty

        # State transition matrix for constant velocity model
        self.A = np.eye(6)
        self.A[0, 3] = dt
        self.A[1, 4] = dt
        self.A[2, 5] = dt

        # Process noise covariance (tune these values!)
        # Higher values trust the model less, measurements more
        pos_process_noise = 0.001 # Noise in assumed constant velocity (m^2/s^2 ? units are tricky)
        vel_process_noise = 0.01
        self.Q = np.diag([pos_process_noise]*3 + [vel_process_noise]*3)

        # Measurement matrix: we measure position only
        self.H = np.hstack((np.eye(3), np.zeros((3, 3))))

        # Measurement noise covariances (tune these values!)
        # Lower values trust the measurement more
        self.R_pose = np.eye(3) * 0.005**2   # std dev of 5mm for pose estimation
        self.R_depth = np.eye(3) * 0.01**2   # std dev of 10mm for direct depth measurement

    def predict(self, dt):
        # Update state transition matrix with current dt
        self.A[0, 3] = dt
        self.A[1, 4] = dt
        self.A[2, 5] = dt

        # Predict the state and covariance
        # Handle potential NaNs or Infs in state
        if not np.all(np.isfinite(self.x)):
            print("Warning: Non-finite state before prediction. Resetting velocity?")
            self.x[3:] = 0 # Reset velocity if state is bad
            if not np.all(np.isfinite(self.x)): # If position also bad, can't do much
                 print("Error: Non-finite position state. Cannot predict.")
                 return # Or reset position to zero? Risky.

        self.x = self.A @ self.x
        self.P = self.A @ self.P @ self.A.T + self.Q
        # Ensure P remains symmetric and positive semi-definite (optional robustness)
        self.P = (self.P + self.P.T) / 2.0

    def update(self, z, R):
        """
        Update the filter with measurement z (3x1 vector) and measurement noise covariance R.
        """
        if z is None or not np.all(np.isfinite(z)):
             # print("Warning: Invalid measurement received. Skipping update.")
             return # Skip update if measurement is invalid

        z = z.reshape((3, 1))

        # Handle potential NaNs or Infs in predicted state
        if not np.all(np.isfinite(self.x)):
             print("Warning: Non-finite state before update. Cannot update.")
             return

        try:
            y = z - self.H @ self.x                   # measurement residual
            S = self.H @ self.P @ self.H.T + R        # residual covariance
            # Ensure S is invertible, add small epsilon if needed
            S = S + np.eye(S.shape[0]) * 1e-9
            K = self.P @ self.H.T @ np.linalg.inv(S)  # Kalman gain

            # Check K for validity
            if not np.all(np.isfinite(K)):
                 print("Warning: Kalman gain is non-finite. Skipping update.")
                 return

            self.x = self.x + K @ y
            self.P = (np.eye(6) - K @ self.H) @ self.P
             # Ensure P remains symmetric and positive semi-definite
            self.P = (self.P + self.P.T) / 2.0

            # Check for explosion of state or covariance
            if np.any(np.abs(self.x) > 10.0): # Position > 10m or velocity > 10m/s likely wrong
                 print(f"Warning: Filter state might be diverging: {self.x.flatten()}")
            if np.any(np.diag(self.P) > 1.0): # High variance
                 print(f"Warning: Filter covariance might be diverging (diag): {np.diag(self.P)}")


        except np.linalg.LinAlgError:
            print("Warning: Singular matrix S in Kalman update. Skipping update.")
        except Exception as e:
             print(f"Error during Kalman update: {e}")

#######################################
# Transformation Helper Functions (Unchanged)
#######################################
def isRotationMatrix(R): # ... (same as before)
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6
def rotationMatrixToEulerAngles(R): # ... (same as before)
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
    return np.array([math.degrees(x), math.degrees(y), math.degrees(z)])
def eulerAnglesToRotationMatrix(theta): # ... (same as before)
    theta_rad = np.radians(theta)
    R_x = np.array([[1,0,0],[0,math.cos(theta_rad[0]),-math.sin(theta_rad[0])],[0,math.sin(theta_rad[0]),math.cos(theta_rad[0])]])
    R_y = np.array([[math.cos(theta_rad[1]),0,math.sin(theta_rad[1])],[0,1,0],[-math.sin(theta_rad[1]),0,math.cos(theta_rad[1])]])
    R_z = np.array([[math.cos(theta_rad[2]),-math.sin(theta_rad[2]),0],[math.sin(theta_rad[2]),math.cos(theta_rad[2]),0],[0,0,1]])
    R = np.dot(R_z, np.dot(R_y, R_x))
    return R
def build_transform(rvec, tvec): # ... (same as before)
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = R
    T[:3, 3] = tvec.reshape(3)
    return T
def invert_transform(T): # ... (same as before)
    R = T[:3, :3]
    t = T[:3, 3]
    R_inv = R.T
    t_inv = -R_inv @ t
    T_inv = np.eye(4, dtype=np.float32)
    T_inv[:3, :3] = R_inv
    T_inv[:3, 3] = t_inv
    return T_inv
# build_transform_from_offset_dict not needed if using separate pos/ori
def decompose_transform(T): # ... (same as before)
    x = T[0, 3]
    y = T[1, 3]
    z = T[2, 3]
    R = T[:3, :3]
    thetas = rotationMatrixToEulerAngles(R)
    theta_x = thetas[0]
    theta_y = thetas[1]
    theta_z = thetas[2]
    return x, y, z, theta_x, theta_y, theta_z

#######################################
# Robot Action Helper Functions (Unchanged)
#######################################
def check_for_end_or_abort(e): # ... (same as before)
    def check(notification, e=e):
        # print("EVENT: " + Base_pb2.ActionEvent.Name(notification.action_event)) # Less verbose
        if (notification.action_event == Base_pb2.ACTION_END or
            notification.action_event == Base_pb2.ACTION_ABORT):
            e.set()
    return check
def populateCartesianCoordinate(waypointInformation): # ... (same as before)
    waypoint = Base_pb2.CartesianWaypoint()
    waypoint.pose.x = waypointInformation[0]
    waypoint.pose.y = waypointInformation[1]
    waypoint.pose.z = waypointInformation[2]
    waypoint.pose.theta_x = waypointInformation[3]
    waypoint.pose.theta_y = waypointInformation[4]
    waypoint.pose.theta_z = waypointInformation[5]
    waypoint.blending_radius = waypointInformation[6]
    waypoint.reference_frame = Base_pb2.CARTESIAN_REFERENCE_FRAME_BASE
    return waypoint
def open_gripper(base): # ... (same as before)
    gripper_command = Base_pb2.GripperCommand()
    finger = gripper_command.gripper.finger.add()
    gripper_command.mode = Base_pb2.GRIPPER_POSITION
    finger.finger_identifier = 1
    finger.value = 0.0
    print("Opening gripper...")
    base.SendGripperCommand(gripper_command)
    time.sleep(1.5)
def close_gripper(base): # ... (same as before)
    gripper_command = Base_pb2.GripperCommand()
    finger = gripper_command.gripper.finger.add()
    gripper_command.mode = Base_pb2.GRIPPER_POSITION
    finger.finger_identifier = 1
    finger.value = 0.7
    print("Closing gripper...")
    base.SendGripperCommand(gripper_command)
    time.sleep(1.5)
def example_move_to_home_position(base): # ... (same as before)
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
    if action_handle is None: return False # Simplified error handling
    e = threading.Event()
    notification_handle = base.OnNotificationActionTopic(check_for_end_or_abort(e), Base_pb2.NotificationOptions())
    base.ExecuteActionFromReference(action_handle)
    finished = e.wait(TIMEOUT_DURATION)
    base.Unsubscribe(notification_handle)
    print("Home position reached." if finished else "Timeout waiting for Home position.")
    return finished

#######################################
# Camera & Detection Function (MODIFIED)
#######################################
def detect_markers_and_filter(marker_size, required_marker_ids, base_marker_id, pick_marker_id, T_base_marker_known, filter_duration_sec):
    """
    Detects markers, calculates camera pose, refines pick marker pose with Kalman filter,
    and returns necessary info.

    Args:
        marker_size: Size in meters.
        required_marker_ids: List of marker IDs needed for the task (e.g., [0, 1]).
        base_marker_id: ID for camera localization.
        pick_marker_id: ID of the marker to be filtered (e.g., 0).
        T_base_marker_known: Known 4x4 transform of the base marker.
        filter_duration_sec: How long to run the refinement filter.

    Returns:
        tuple: (T_base_camera, detected_markers_camera_raw, filtered_pick_marker_pos_base)
               T_base_camera: 4x4 numpy array (or None).
               detected_markers_camera_raw: Dict {id: T_camera_marker} raw poses (or None).
               filtered_pick_marker_pos_base: Filtered [x,y,z] of pick marker in base frame (or None).
    """
    pipeline = None
    T_base_camera = None
    detected_markers_camera_raw = None
    filtered_pick_marker_pos_base = None
    initial_pick_marker_pos_base = None
    kf = None # Kalman Filter instance

    # --- Phase 1: Initial Detection ---
    print("--- Detection Phase 1: Initial Scan ---")
    try:
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        profile = pipeline.start(config)
        align = rs.align(rs.stream.color)
        print("RealSense pipeline started.")

        color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
        intrinsics = color_profile.get_intrinsics()
        camera_matrix = np.array([[intrinsics.fx, 0, intrinsics.ppx], [0, intrinsics.fy, intrinsics.ppy], [0, 0, 1]])
        dist_coeffs = np.array(intrinsics.coeffs)

        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
        aruco_params = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

        start_time = time.time()
        detection_timeout = 10 # seconds

        while time.time() - start_time < detection_timeout:
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            color_frame = aligned_frames.get_color_frame()
            if not color_frame: continue

            color_image = np.asanyarray(color_frame.get_data())
            gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = detector.detectMarkers(gray)

            current_detections_camera = {}
            T_base_camera_frame = None # Recalculate each frame

            if ids is not None:
                rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, marker_size, camera_matrix, dist_coeffs)
                cv2.aruco.drawDetectedMarkers(color_image, corners, ids)

                for i, marker_id in enumerate(ids.flatten()):
                    T_camera_marker = build_transform(rvecs[i][0], tvecs[i][0])
                    current_detections_camera[marker_id] = T_camera_marker
                    cv2.drawFrameAxes(color_image, camera_matrix, dist_coeffs, rvecs[i][0], tvecs[i][0], marker_size * 0.75)
                    if marker_id == base_marker_id:
                        T_marker_camera = invert_transform(T_camera_marker)
                        T_base_camera_frame = T_base_marker_known @ T_marker_camera

            cv2.imshow("Initial Detection", color_image)
            if cv2.waitKey(1) & 0xFF == ord('q'): return None, None, None

            if T_base_camera_frame is not None:
                all_required_found = all(req_id in current_detections_camera for req_id in required_marker_ids)
                if all_required_found and pick_marker_id in current_detections_camera:
                    print("Initial detection successful. Found base and required markers.")
                    T_base_camera = T_base_camera_frame # Store the last good T_base_camera
                    detected_markers_camera_raw = current_detections_camera # Store raw poses
                    # Calculate initial pick marker pose for filter init
                    T_base_pick_marker_initial = T_base_camera @ detected_markers_camera_raw[pick_marker_id]
                    initial_pick_marker_pos_base = T_base_pick_marker_initial[:3, 3]
                    break # Exit initial detection loop

        cv2.destroyWindow("Initial Detection")
        if T_base_camera is None or detected_markers_camera_raw is None or initial_pick_marker_pos_base is None:
             print("Failed to detect all required markers in Phase 1.")
             if pipeline: pipeline.stop()
             return None, None, None

        # --- Phase 2: Kalman Filter Refinement for Pick Marker ---
        print(f"--- Detection Phase 2: Refining Marker {pick_marker_id} Pose ---")
        if initial_pick_marker_pos_base is not None:
            # Initialize Kalman Filter
            initial_state = np.zeros((6, 1))
            initial_state[:3, 0] = initial_pick_marker_pos_base
            kf = KalmanFilter6D(dt=1.0/30.0, initial_state=initial_state) # dt will be updated
            print(f"Kalman filter initialized for marker {pick_marker_id} at {initial_pick_marker_pos_base.flatten()}")

            start_time = time.time()
            prev_filter_time = start_time
            frame_count = 0

            while time.time() - start_time < filter_duration_sec:
                frames = pipeline.wait_for_frames()
                aligned_frames = align.process(frames)
                depth_frame_f = aligned_frames.get_depth_frame()
                color_frame_f = aligned_frames.get_color_frame()
                if not depth_frame_f or not color_frame_f: continue
                frame_count += 1

                color_image_f = np.asanyarray(color_frame_f.get_data())
                gray_f = cv2.cvtColor(color_image_f, cv2.COLOR_BGR2GRAY)
                corners_f, ids_f, _ = detector.detectMarkers(gray_f)

                # --- Measurements (Recalculated in Base Frame each time) ---
                measurement_pose = None
                measurement_depth = None
                T_base_camera_f = None # Recalculate T_base_camera

                if ids_f is not None:
                    rvecs_f, tvecs_f, _ = cv2.aruco.estimatePoseSingleMarkers(corners_f, marker_size, camera_matrix, dist_coeffs)
                    cv2.aruco.drawDetectedMarkers(color_image_f, corners_f, ids_f)

                    # 1. Find Base Marker to get current T_base_camera_f
                    base_idx = np.where(ids_f.flatten() == base_marker_id)[0]
                    if len(base_idx) > 0:
                         idx = base_idx[0]
                         T_camera_base_marker = build_transform(rvecs_f[idx][0], tvecs_f[idx][0])
                         T_base_camera_f = T_base_marker_known @ invert_transform(T_camera_base_marker)
                         cv2.drawFrameAxes(color_image_f, camera_matrix, dist_coeffs, rvecs_f[idx][0], tvecs_f[idx][0], marker_size * 0.5)


                    # 2. Find Pick Marker and get measurements IF T_base_camera_f is valid
                    pick_idx = np.where(ids_f.flatten() == pick_marker_id)[0]
                    if len(pick_idx) > 0 and T_base_camera_f is not None:
                         idx = pick_idx[0]
                         # Measurement from Pose Estimation
                         T_camera_pick_marker = build_transform(rvecs_f[idx][0], tvecs_f[idx][0])
                         T_base_pick_marker = T_base_camera_f @ T_camera_pick_marker
                         measurement_pose = T_base_pick_marker[:3, 3]
                         cv2.drawFrameAxes(color_image_f, camera_matrix, dist_coeffs, rvecs_f[idx][0], tvecs_f[idx][0], marker_size * 0.75)

                         # Measurement from Direct Depth
                         marker_corners = corners_f[idx].reshape((4, 2))
                         center_x = int(np.mean(marker_corners[:, 0]))
                         center_y = int(np.mean(marker_corners[:, 1]))
                         # Check bounds before getting depth
                         if 0 <= center_x < intrinsics.width and 0 <= center_y < intrinsics.height:
                             depth_value = depth_frame_f.get_distance(center_x, center_y)
                             if depth_value > 0.1: # Check if depth is valid (e.g. > 10cm)
                                 x_cam = (center_x - intrinsics.ppx) * depth_value / intrinsics.fx
                                 y_cam = (center_y - intrinsics.ppy) * depth_value / intrinsics.fy
                                 point_cam = np.array([x_cam, y_cam, depth_value, 1.0], dtype=np.float32)
                                 point_base = T_base_camera_f @ point_cam
                                 measurement_depth = point_base[:3]
                         else:
                             print(f"Warning: Marker center ({center_x},{center_y}) out of bounds.")


                # --- Kalman Filter Step ---
                current_filter_time = time.time()
                dt = current_filter_time - prev_filter_time
                prev_filter_time = current_filter_time

                if kf and dt > 0: # Ensure dt is positive
                    kf.predict(dt)
                    if measurement_pose is not None:
                        kf.update(measurement_pose, kf.R_pose)
                    if measurement_depth is not None:
                        kf.update(measurement_depth, kf.R_depth)

                    filtered_pick_marker_pos_base = kf.x[:3, 0] # Update the result continuously

                    # --- Display Filter Phase ---
                    cv2.putText(color_image_f, f"Pose: {measurement_pose[0]:.3f},{measurement_pose[1]:.3f},{measurement_pose[2]:.3f}" if measurement_pose is not None else "Pose: N/A",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    cv2.putText(color_image_f, f"Depth: {measurement_depth[0]:.3f},{measurement_depth[1]:.3f},{measurement_depth[2]:.3f}" if measurement_depth is not None else "Depth: N/A",
                                (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 0), 1)
                    cv2.putText(color_image_f, f"Fused: {filtered_pick_marker_pos_base[0]:.4f},{filtered_pick_marker_pos_base[1]:.4f},{filtered_pick_marker_pos_base[2]:.4f}",
                                (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    cv2.imshow("Filter Refinement", color_image_f)
                    if cv2.waitKey(1) & 0xFF == ord('q'): return None, None, None

            print(f"Filter refinement finished after {frame_count} frames ({time.time() - start_time:.2f}s).")
            if filtered_pick_marker_pos_base is not None:
                 print(f"Final filtered position for marker {pick_marker_id}: {filtered_pick_marker_pos_base.flatten()}")
            else:
                 print(f"Warning: Filter refinement did not produce a valid position for marker {pick_marker_id}.")

        else:
             print(f"Cannot start filter refinement: Initial position for marker {pick_marker_id} not found.")


    except Exception as e:
        print(f"Error during detection/filtering: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None
    finally:
        if pipeline:
            pipeline.stop()
            print("RealSense pipeline stopped.")
        cv2.destroyAllWindows()
        print("Detection windows closed.")

    # Return the results (raw poses are from the *end* of Phase 1)
    return T_base_camera, detected_markers_camera_raw, filtered_pick_marker_pos_base


#######################################
# Motion Execution Functions (MODIFIED calculate_target_pose)
#######################################

def calculate_target_pose_in_base(ref_marker_id, position_offset_dict, target_orientation_base_dict,
                                  T_base_camera, detected_markers_camera_raw,
                                  filtered_marker_positions_base=None):
    """
    Calculates the target pose in the robot base frame.
    Position is relative to the marker (using filtered position if available).
    Orientation is specified in the base frame.
    """
    if T_base_camera is None:
        print("Error: Cannot calculate target pose, T_base_camera is None.")
        return None
    if ref_marker_id not in detected_markers_camera_raw:
        print(f"Error: Cannot calculate target pose, reference marker {ref_marker_id} raw pose not found.")
        return None

    # --- Determine the reference marker's pose in the base frame ---
    T_base_marker_ref = None
    use_filtered = False
    if filtered_marker_positions_base and ref_marker_id in filtered_marker_positions_base:
        filtered_pos = filtered_marker_positions_base[ref_marker_id]
        if filtered_pos is not None and np.all(np.isfinite(filtered_pos)):
            # Use filtered POSITION + raw ORIENTATION
            T_camera_marker_ref_raw = detected_markers_camera_raw[ref_marker_id]
            T_base_marker_ref_raw = T_base_camera @ T_camera_marker_ref_raw
            R_base_marker_ref_raw = T_base_marker_ref_raw[:3, :3] # Get orientation from raw

            T_base_marker_ref = np.eye(4, dtype=np.float32)
            T_base_marker_ref[:3, :3] = R_base_marker_ref_raw
            T_base_marker_ref[:3, 3] = filtered_pos.flatten() # Use filtered position
            use_filtered = True
            # print(f"Using FILTERED position for marker {ref_marker_id}")
        else:
            print(f"Warning: Filtered position for marker {ref_marker_id} is invalid. Using raw pose.")

    if T_base_marker_ref is None: # If not using filtered or filtered was invalid
        # Use raw pose
        T_camera_marker_ref_raw = detected_markers_camera_raw[ref_marker_id]
        T_base_marker_ref = T_base_camera @ T_camera_marker_ref_raw
        # print(f"Using RAW pose for marker {ref_marker_id}")


    # --- Calculate final target pose ---
    # Create purely positional offset transform relative to the (potentially filtered) marker frame
    T_marker_pos_offset = np.eye(4, dtype=np.float32)
    T_marker_pos_offset[0, 3] = position_offset_dict['x']
    T_marker_pos_offset[1, 3] = position_offset_dict['y']
    T_marker_pos_offset[2, 3] = position_offset_dict['z']

    # Calculate the final target position in the base frame
    T_base_target_pos_only = T_base_marker_ref @ T_marker_pos_offset
    P_base_target = T_base_target_pos_only[:3, 3]

    # Create the target rotation matrix from the desired base frame orientation
    R_base_target = eulerAnglesToRotationMatrix([
        target_orientation_base_dict['theta_x'],
        target_orientation_base_dict['theta_y'],
        target_orientation_base_dict['theta_z']
    ])

    # Combine the target position and target rotation
    T_base_target = np.eye(4, dtype=np.float32)
    T_base_target[:3, :3] = R_base_target
    T_base_target[:3, 3] = P_base_target

    # print(f"Calculated T_base_target (filtered={use_filtered}):\n{T_base_target}")
    return T_base_target


def move_to_pose_relative_to_marker(base, ref_marker_id, position_offset_dict, target_orientation_base_dict,
                                     T_base_camera, detected_markers_camera_raw,
                                     filtered_marker_positions_base=None, # Added optional arg
                                     blending_radius=0.0):
    """Moves robot using filtered position for ref_marker_id if available."""
    print(f"\nAttempting move relative to Marker ID: {ref_marker_id}")
    # print(f"Position Offset: {position_offset_dict}")
    # print(f"Target Base Orientation: {target_orientation_base_dict}")

    # Call the modified calculation function, passing filtered data
    T_base_target = calculate_target_pose_in_base(ref_marker_id, position_offset_dict, target_orientation_base_dict,
                                                  T_base_camera, detected_markers_camera_raw,
                                                  filtered_marker_positions_base) # Pass it down

    if T_base_target is None:
        print("Move failed: Could not calculate target pose.")
        return False

    try:
        x, y, z, theta_x, theta_y, theta_z = decompose_transform(T_base_target)
        print(f"Target Base Coordinates: x={x:.4f}, y={y:.4f}, z={z:.4f}, tx={theta_x:.1f}, ty={theta_y:.1f}, tz={theta_z:.1f}") # Increased precision

        waypoint_info = (x, y, z, theta_x, theta_y, theta_z, blending_radius)
        waypoints = Base_pb2.WaypointList()
        waypoints.duration = 0.0
        waypoints.use_optimal_blending = False
        waypoint = waypoints.waypoints.add()
        waypoint.name = f"move_rel_marker_{ref_marker_id}_base_ori_filt"
        waypoint.cartesian_waypoint.CopyFrom(populateCartesianCoordinate(waypoint_info))

        # print("Validating trajectory...") # Less verbose
        result = base.ValidateWaypointList(waypoints)
        if len(result.trajectory_error_report.trajectory_error_elements) != 0:
            print("FATAL: Trajectory validation failed:") ; print(result.trajectory_error_report)
            return False
        # print("Trajectory valid.")

        e = threading.Event()
        notification_handle = base.OnNotificationActionTopic(check_for_end_or_abort(e), Base_pb2.NotificationOptions())

        # print("Executing trajectory...") # Less verbose
        base.ExecuteWaypointTrajectory(waypoints)
        finished = e.wait(TIMEOUT_DURATION)
        base.Unsubscribe(notification_handle)

        print("Move successful." if finished else "Move timed out.")
        return finished

    except Exception as e:
        print(f"Error during move execution: {e}")
        return False


#######################################
# Main Function (MODIFIED)
#######################################
def main():
    # Import the utilities helper module
    # Assumes utilities.py is in the parent directory
    script_dir = os.path.dirname(__file__)
    parent_dir = os.path.abspath(os.path.join(script_dir, '..'))
    if parent_dir not in sys.path:
         sys.path.insert(0, parent_dir)
    try:
        import utilities
    except ImportError:
        print("ERROR: Failed to import 'utilities' module.")
        print("Ensure 'utilities.py' from Kinova examples is in the parent directory or adjust sys.path.")
        return 1

    # Parse arguments
    args = utilities.parseConnectionArguments()
    if args is None: # Handle case where parsing fails (e.g., user cancels)
         return 1


    # Create connection to the device and get the router
    with utilities.DeviceConnection.createTcpConnection(args) as router:
        base = BaseClient(router)
        success = True
        T_base_camera = None
        detected_markers_raw = None
        filtered_pick_pos = None

        try:
            # 1. Move Home
            success &= example_move_to_home_position(base)
            if not success: raise RuntimeError("Failed to reach home position.")

            # 2. Open Gripper
            open_gripper(base)

            # 3. Detect Markers & Filter Pick Marker
            print("\n--- Starting Marker Detection & Filtering ---")
            required_markers = [PICK_MARKER_ID, PLACE_MARKER_ID]
            # Call the new detection function
            T_base_camera, detected_markers_raw, filtered_pick_pos = detect_markers_and_filter(
                MARKER_SIZE, required_markers, BASE_MARKER_ID, PICK_MARKER_ID, T_base_marker, FILTER_DURATION
            )

            if T_base_camera is None or detected_markers_raw is None:
                 raise RuntimeError("Failed detection phase. Exiting.")
            if filtered_pick_pos is None:
                 print("Warning: Filtering failed for pick marker, proceeding with raw pose if possible.")
            print("--- Detection & Filtering Complete ---")

            # Prepare filtered positions dictionary for motion function
            filtered_positions = {}
            if filtered_pick_pos is not None:
                 filtered_positions[PICK_MARKER_ID] = filtered_pick_pos


            # --- Pick Sequence (Uses Filtered Position for Marker 0) ---
            print("\n--- Starting Pick Sequence ---")
            # 4. Approach Pick Marker
            success &= move_to_pose_relative_to_marker(base, PICK_MARKER_ID, PICK_APPROACH_POS_OFFSET, TARGET_ORIENTATION_BASE,
                                                      T_base_camera, detected_markers_raw, filtered_positions)
            if not success: raise RuntimeError("Failed pick approach move.")

            # 5. Descend to Pick Marker
            success &= move_to_pose_relative_to_marker(base, PICK_MARKER_ID, PICK_GRASP_POS_OFFSET, TARGET_ORIENTATION_BASE,
                                                      T_base_camera, detected_markers_raw, filtered_positions)
            if not success: raise RuntimeError("Failed pick descend move.")

            # 6. Close Gripper
            close_gripper(base)
            time.sleep(1.0)

            # 7. Ascend from Pick Marker
            success &= move_to_pose_relative_to_marker(base, PICK_MARKER_ID, PICK_APPROACH_POS_OFFSET, TARGET_ORIENTATION_BASE,
                                                      T_base_camera, detected_markers_raw, filtered_positions)
            if not success: raise RuntimeError("Failed pick ascend move.")
            print("--- Pick Sequence Complete ---")


            # --- Place Sequence (Uses Raw Position for Marker 1) ---
            print("\n--- Starting Place Sequence ---")
            # 8. Approach Place Marker
            success &= move_to_pose_relative_to_marker(base, PLACE_MARKER_ID, PLACE_APPROACH_POS_OFFSET, TARGET_ORIENTATION_BASE,
                                                      T_base_camera, detected_markers_raw, filtered_positions) # Pass filtered dict, but it won't be used for ID 1
            if not success: raise RuntimeError("Failed place approach move.")

            # 9. Descend to Place Marker
            success &= move_to_pose_relative_to_marker(base, PLACE_MARKER_ID, PLACE_RELEASE_POS_OFFSET, TARGET_ORIENTATION_BASE,
                                                      T_base_camera, detected_markers_raw, filtered_positions)
            if not success: raise RuntimeError("Failed place descend move.")

            # 10. Open Gripper
            # open_gripper(base)
            time.sleep(2.0)

            # 11. Ascend from Place Marker
            success &= move_to_pose_relative_to_marker(base, PLACE_MARKER_ID, PLACE_APPROACH_POS_OFFSET, TARGET_ORIENTATION_BASE,
                                                      T_base_camera, detected_markers_raw, filtered_positions)
            if not success: raise RuntimeError("Failed place ascend move.")
            print("--- Place Sequence Complete ---")




            # 12. Move back Home
            print("\n--- Returning to Home ---")
            success &= example_move_to_home_position(base)

        except Exception as e:
            print(f"\n!!!!!!!!!!!!!!!!!!!!!!!!!!!\nAn error occurred: {e}\n!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            import traceback
            traceback.print_exc()
            success = False
        finally:
            if 'base' in locals() and base is not None:
                 print("Ensuring arm is home...")
                 example_move_to_home_position(base) # Attempt homing on exit/error

        return 0 if success else 1

if __name__ == "__main__":
    # Add robustness for finding utilities.py
    try:
        # Attempt to run main
        exit_code = main()
        sys.exit(exit_code)
    except ImportError as e:
        print(f"Import Error: {e}")
        print("Please ensure 'utilities.py' is accessible.")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred in __main__: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)