import os
import sys
import time
import threading
import numpy as np
import cv2
import pyrealsense2 as rs
import math
import requests
import websocket
import json
import rel
import ssl
import utilities
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv('API_KEY')
API_ID = os.getenv('API_ID')

# Manipulator API Imports
from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
from kortex_api.autogen.messages import Base_pb2, BaseCyclic_pb2, Common_pb2

ROBOT_NUMBER = 1 # set as appropriate
API_URL = 'https://roboticsensingproject.org:4001/api/submitRobotStatus'
FULLCHAIN_PATH = './test.pem'
CA_PATH = './test.pem' # Path to CA cert for WSS verification
WEBSOCKET_URI = "wss://roboticsensingproject.org:4001"

# Task Parameters
TIMEOUT_DURATION = 65
PICK_MARKER_ID = 3
BASE_MARKER_ID = 8
MARKER_SIZE = 0.04 # meters
FILTER_DURATION = 1.5 # seconds to run filter refinement
HOLD_DURATION = 4.0 # seconds to wait at placement location

# --- Pick Positional Offsets relative to marker position IN BASE FRAME ---
# Approach 10cm above the marker's detected position
PICK_APPROACH_POS_OFFSET = {'x': -0.13, 'y': -0.02, 'z': 0.04}
# Grasp 1cm above the marker's detected position
PICK_GRASP_POS_OFFSET    = {'x': -0.02, 'y': -0.02, 'z': -0.02}
PICK_LIFT_POS_OFFSET = {'x': 0.0, 'y': -0.02, 'z': 0.25} # Lift 5cm above the grasp position

# --- Target Orientation in Base Frame ---
TARGET_ORIENTATION_BASE = {'theta_x': 90.0, 'theta_y': 0.0, 'theta_z': 90.0}

# --- Placement Target Definitions (Redefine offsets in BASE FRAME) ---
PLACE_TARGETS = {
    "pot1": {
        "marker_id": 0,
        # Approach 25cm directly above marker 0's detected position
        "approach_offset": {'x': 0.0, 'y': 0.0, 'z': 0.3},
        # Hold/Place 5cm directly above marker 0's detected position
        "hold_offset": {'x': 0.0, 'y': 0.0, 'z': 0.15}
    },
    "pot2": {
        "marker_id": 2,
        # Approach 25cm directly above marker 2's detected position
        "approach_offset": {'x': 0.0, 'y': 0.0, 'z': 0.3},
        # Hold/Place 5cm directly above marker 2's detected position
        "hold_offset": {'x': 0.0, 'y': 0.0, 'z': 0.15}
    },
    "pot3": {
        "marker_id": 4,
        # Approach 25cm directly above marker 2's detected position
        "approach_offset": {'x': 0.0, 'y': 0.0, 'z': 0.15},
        # Hold/Place 5cm directly above marker 2's detected position
        "hold_offset": {'x': 0.0, 'y': 0.0, 'z': 0.05}
    },
}

# --- SELECT WHICH TARGET TO USE FOR DFAULT RUN ---
DEFAULT_PLACE_TARGET_KEY = "pot2"
# ---

# Known transform for the base marker from Robot base frame (ID 8)
T_base_marker = np.array([
    [1, 0, 0, -0.09],
    [0, 1, 0,  0.0],
    [0, 0, 1,  0.0],
    [0, 0, 0,  1.0]
], dtype=np.float32)


#######################################
# Kalman Filter Class 
#######################################
class KalmanFilter6D:
    def __init__(self, dt, initial_state):
        # State vector: [x, y, z, vx, vy, vz]^T (shape 6,1)
        self.x = initial_state.reshape(6,1) if initial_state is not None else np.zeros((6, 1))
        self.P = np.eye(6) * 1.0
        self.A = np.eye(6)
        self.A[0, 3] = dt; self.A[1, 4] = dt; self.A[2, 5] = dt 
        pos_process_noise = 0.001; vel_process_noise = 0.01
        self.Q = np.diag([pos_process_noise]*3 + [vel_process_noise]*3)
        self.H = np.hstack((np.eye(3), np.zeros((3, 3))))
        self.R_pose = np.eye(3) * (0.015**2) # standard deviation of 1.5cm for pose measurement from aruco 
        self.R_depth = np.eye(3) * (0.01**2) # standard deviation of 1cm for depth measurement from depth camera
    def predict(self, dt):
        self.A[0, 3] = dt; self.A[1, 4] = dt; self.A[2, 5] = dt
        if not np.all(np.isfinite(self.x)):
            print("Warning: Non-finite state before prediction. Resetting velocity?")
            self.x[3:] = 0
            if not np.all(np.isfinite(self.x)): return
        self.x = self.A @ self.x
        self.P = self.A @ self.P @ self.A.T + self.Q
        self.P = (self.P + self.P.T) / 2.0
    def update(self, z, R):
        if z is None or not np.all(np.isfinite(z)): return
        z = z.reshape((3, 1))
        if not np.all(np.isfinite(self.x)): return
        try:
            y = z - self.H @ self.x
            S = self.H @ self.P @ self.H.T + R
            S = S + np.eye(S.shape[0]) * 1e-9
            K = self.P @ self.H.T @ np.linalg.inv(S)
            if not np.all(np.isfinite(K)): return
            self.x = self.x + K @ y
            self.P = (np.eye(6) - K @ self.H) @ self.P
            self.P = (self.P + self.P.T) / 2.0
            # divergence checks...
        except np.linalg.LinAlgError: pass # print("Warning: Singular matrix S...")
        except Exception as e: print(f"Error during Kalman update: {e}")


#######################################
# Transformation Helper Functions
#######################################

def isRotationMatrix(R): # ...
    Rt = np.transpose(R); shouldBeIdentity = np.dot(Rt, R); I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity); return n < 1e-6
def rotationMatrixToEulerAngles(R): # ...
    assert (isRotationMatrix(R)); sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6
    if not singular: x = math.atan2(R[2, 1], R[2, 2]); y = math.atan2(-R[2, 0], sy); z = math.atan2(R[1, 0], R[0, 0])
    else: x = math.atan2(-R[1, 2], R[1, 1]); y = math.atan2(-R[2, 0], sy); z = 0
    return np.array([math.degrees(x), math.degrees(y), math.degrees(z)])
def eulerAnglesToRotationMatrix(theta): # ...
    theta_rad=np.radians(theta);R_x=np.array([[1,0,0],[0,math.cos(theta_rad[0]),-math.sin(theta_rad[0])],[0,math.sin(theta_rad[0]),math.cos(theta_rad[0])]]);R_y=np.array([[math.cos(theta_rad[1]),0,math.sin(theta_rad[1])],[0,1,0],[-math.sin(theta_rad[1]),0,math.cos(theta_rad[1])]]);R_z=np.array([[math.cos(theta_rad[2]),-math.sin(theta_rad[2]),0],[math.sin(theta_rad[2]),math.cos(theta_rad[2]),0],[0,0,1]]);R=np.dot(R_z,np.dot(R_y,R_x));return R
def build_transform(rvec, tvec): # ...
    R,_=cv2.Rodrigues(rvec);T=np.eye(4,dtype=np.float32);T[:3,:3]=R;T[:3,3]=tvec.reshape(3);return T
def invert_transform(T): # ...
    R=T[:3,:3];t=T[:3,3];R_inv=R.T;t_inv=-R_inv@t;T_inv=np.eye(4,dtype=np.float32);T_inv[:3,:3]=R_inv;T_inv[:3,3]=t_inv;return T_inv
def decompose_transform(T): # ...
    x=T[0,3];y=T[1,3];z=T[2,3];R=T[:3,:3];thetas=rotationMatrixToEulerAngles(R);theta_x=thetas[0];theta_y=thetas[1];theta_z=thetas[2];return x,y,z,theta_x,theta_y,theta_z


#######################################
# Robot Action Helper Functions 
#######################################

def check_for_end_or_abort(e): # ...
    def check(notification,e=e): #print("EVENT:"+Base_pb2.ActionEvent.Name(notification.action_event))
        if(notification.action_event==Base_pb2.ACTION_END or notification.action_event==Base_pb2.ACTION_ABORT): e.set()
    return check
def populateCartesianCoordinate(waypointInformation): # ...
    waypoint=Base_pb2.CartesianWaypoint()
    waypoint.pose.x=waypointInformation[0]
    waypoint.pose.y=waypointInformation[1]
    waypoint.pose.z=waypointInformation[2]
    waypoint.pose.theta_x=waypointInformation[3]
    waypoint.pose.theta_y=waypointInformation[4]
    waypoint.pose.theta_z=waypointInformation[5]
    waypoint.blending_radius=waypointInformation[6]
    waypoint.reference_frame=Base_pb2.CARTESIAN_REFERENCE_FRAME_BASE
    return waypoint
def open_gripper(base): # ...
    gripper_command=Base_pb2.GripperCommand()
    finger=gripper_command.gripper.finger.add()
    gripper_command.mode=Base_pb2.GRIPPER_POSITION;
    finger.finger_identifier=1;finger.value=0.0
    print("Opening gripper...")
    base.SendGripperCommand(gripper_command)
    time.sleep(1.5)
def close_gripper(base): # ...
    gripper_command=Base_pb2.GripperCommand()
    finger=gripper_command.gripper.finger.add()
    gripper_command.mode=Base_pb2.GRIPPER_POSITION
    finger.finger_identifier=1
    finger.value=0.7
    print("Closing gripper...")
    base.SendGripperCommand(gripper_command)
    time.sleep(1.5)
def example_move_to_home_position(base): # ...
    base_servo_mode=Base_pb2.ServoingModeInformation()
    base_servo_mode.servoing_mode=Base_pb2.SINGLE_LEVEL_SERVOING
    base.SetServoingMode(base_servo_mode)
    print("Moving the arm to Home position...")
    action_type=Base_pb2.RequestedActionType()
    action_type.action_type=Base_pb2.REACH_JOINT_ANGLES
    action_list=base.ReadAllActions(action_type)
    action_handle=None;
    for action in action_list.action_list:
        if action.name=="Home": 
            action_handle=action.handle
            break
    if action_handle is None: return False
    e=threading.Event()
    notification_handle=base.OnNotificationActionTopic(check_for_end_or_abort(e),Base_pb2.NotificationOptions())
    base.ExecuteActionFromReference(action_handle)
    finished=e.wait(TIMEOUT_DURATION);base.Unsubscribe(notification_handle)
    print("Home position reached."if finished else "Timeout waiting for Home position.")
    return finished




#######################################
# Status Reporting Function 
#######################################
def send_status_update(data):
    """Sends a status update to the API endpoint."""
    if not API_ID or not API_KEY:
        # print("Warning: API_ID or API_KEY not set. Skipping status update.")
        return # Don't try to send if credentials aren't set

    try:
        # Ensure data includes credentials just before sending
        data_with_creds = data.copy()
        data_with_creds["id"] = API_ID
        data_with_creds["key"] = API_KEY
        json_obj = data_with_creds

        # Check if the certificate file exists
        if not os.path.exists(FULLCHAIN_PATH):
            print(f"Error: Certificate file not found at {FULLCHAIN_PATH}. Sending without verification.")
            verify_path = False # Send without verification if cert is missing
        else:
            verify_path = FULLCHAIN_PATH

        response = requests.post(API_URL, verify=verify_path, json=json_obj, timeout=5.0) # Added timeout

        # Optional: Be less verbose on success to avoid cluttering robot logs
        # if response.status_code == 200:
        #     print("Successfully sent status update")
        if response.status_code != 200:
            print(f"Status Update Error: {response.status_code}")
            # print(f"Content: {response.text}") # Maybe only print content on repeated errors
    except requests.exceptions.Timeout:
        print(f"Status Update Error: Request timed out to {API_URL}")
    except requests.exceptions.RequestException as e:
        print(f"Status Update Error: {e}")
    except Exception as e:
        print(f"Status Update Error: An unexpected error occurred - {e}")


#######################################
# Status Reporting Thread Worker
#######################################
def status_reporting_worker(stop_event, shared_status_list, start_time):
    """
    Runs in a separate thread, periodically sending status updates.
    Args:
        stop_event (threading.Event): Event to signal when the thread should stop.
        shared_status_list (list): A list containing the current status string (mutable).
        start_time (float): The time the main task started.
    """
    print("Status reporting thread started.")
    while not stop_event.is_set():
        current_task_message = "Idle"
        if shared_status_list: # Check if list is not empty
            current_task_message = shared_status_list[0] # Read current status

        payload = {
            "robot_number": ROBOT_NUMBER,
            "runtime": round(time.time() - start_time),
            "currentTask": current_task_message
            # ID and Key are added within send_status_update
        }
        send_status_update(payload)

        # Wait for 10 seconds OR until stop_event is set
        stop_event.wait(10.0)
    print("Status reporting thread stopped.")


#######################################
# Camera & Detection Functions
#######################################

def detect_markers_and_filter(marker_size, required_marker_ids, base_marker_id, pick_marker_id, place_marker_id, T_base_marker_known, filter_duration_sec):
    pipeline = None
    T_base_camera = None
    detected_markers_camera_raw = None
    # Dictionary to hold filtered positions ---
    filtered_positions_base = {} # Will store {marker_id: filtered_position}

    # --- Kalman Filters (one for pick, one for place) ---
    kf_pick = None
    kf_place = None
    initial_pick_marker_pos_base = None
    initial_place_marker_pos_base = None

    print("--- Detection Phase 1: Initial Scan ---")
    try:
        # --- Setup RealSense, ArUco detector ---
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
        dist_coeffs = np.array(intrinsics.coeffs) if intrinsics.coeffs else np.zeros(5) # Handle no distortion case
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
        aruco_params = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

        # --- Initial Detection Loop (Find all markers, get initial poses) ---
        start_time = time.time()
        detection_timeout = 10 # seconds
        temp_detected_markers = {} # Store detections from this phase

        while time.time() - start_time < detection_timeout:
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            color_frame = aligned_frames.get_color_frame()
            if not color_frame: continue

            color_image = np.asanyarray(color_frame.get_data())
            gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = detector.detectMarkers(gray)

            current_detections_camera = {}
            T_base_camera_frame = None

            if ids is not None:
                rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, marker_size, camera_matrix, dist_coeffs)
                cv2.aruco.drawDetectedMarkers(color_image, corners, ids)
                for i, marker_id in enumerate(ids.flatten()):
                    if marker_id in required_marker_ids or marker_id == base_marker_id: # Only process required/base markers
                        T_camera_marker = build_transform(rvecs[i][0], tvecs[i][0])
                        current_detections_camera[marker_id] = T_camera_marker
                        cv2.drawFrameAxes(color_image, camera_matrix, dist_coeffs, rvecs[i][0], tvecs[i][0], marker_size * 0.75)
                        if marker_id == base_marker_id:
                            T_marker_camera = invert_transform(T_camera_marker)
                            T_base_camera_frame = T_base_marker_known @ T_marker_camera

            cv2.imshow("Initial Detection", color_image)
            if cv2.waitKey(1) & 0xFF == ord('q'): return None, None, None # Allow quitting

            # --- Check if requirements met ---
            if T_base_camera_frame is not None:
                # Check if all explicitly required markers are found in this frame
                all_required_found = all(req_id in current_detections_camera for req_id in required_marker_ids)
                # Also ensure the base marker is found (implicitly required)
                base_found = base_marker_id in current_detections_camera

                if all_required_found and base_found:
                    print("Initial detection successful. Found base and all required markers.")
                    T_base_camera = T_base_camera_frame # Lock in the camera pose
                    temp_detected_markers = current_detections_camera # Store the raw poses from this frame

                    # Calculate initial base positions for filtering
                    if pick_marker_id in temp_detected_markers:
                        T_base_pick_marker_initial = T_base_camera @ temp_detected_markers[pick_marker_id]
                        initial_pick_marker_pos_base = T_base_pick_marker_initial[:3, 3]
                    if place_marker_id in temp_detected_markers:
                        T_base_place_marker_initial = T_base_camera @ temp_detected_markers[place_marker_id]
                        initial_place_marker_pos_base = T_base_place_marker_initial[:3, 3]
                    break # Exit loop once requirements met

        cv2.destroyWindow("Initial Detection")

        # --- Check if Initial Phase Succeeded ---
        if T_base_camera is None or not temp_detected_markers:
             print("Failed to detect base marker and all required markers in Phase 1.")
             if pipeline: pipeline.stop()
             return None, None, None # Return None for all if initial fails

        # Store the raw poses found during the successful initial detection frame
        detected_markers_camera_raw = temp_detected_markers

        # --- Detection Phase 2: Refining Poses with Kalman Filter ---
        print(f"--- Detection Phase 2: Refining Marker {pick_marker_id} and {place_marker_id} Poses ---")

        # Initialize Kalman Filters if initial positions were found
        if initial_pick_marker_pos_base is not None and np.all(np.isfinite(initial_pick_marker_pos_base)):
            initial_state_pick = np.zeros((6, 1))
            initial_state_pick[:3, 0] = initial_pick_marker_pos_base.flatten()
            kf_pick = KalmanFilter6D(dt=1.0 / 30.0, initial_state=initial_state_pick)
            print(f"Kalman filter initialized for PICK marker {pick_marker_id} at {initial_pick_marker_pos_base.flatten()}")
        else:
            print(f"Warning: Could not get initial position for PICK marker {pick_marker_id}. Filtering disabled for it.")

        if initial_place_marker_pos_base is not None and np.all(np.isfinite(initial_place_marker_pos_base)):
             initial_state_place = np.zeros((6, 1))
             initial_state_place[:3, 0] = initial_place_marker_pos_base.flatten()
             kf_place = KalmanFilter6D(dt=1.0 / 30.0, initial_state=initial_state_place)
             print(f"Kalman filter initialized for PLACE marker {place_marker_id} at {initial_place_marker_pos_base.flatten()}")
        else:
             print(f"Warning: Could not get initial position for PLACE marker {place_marker_id}. Filtering disabled for it.")

        # --- Filtering Loop ---
        if kf_pick or kf_place: # Only run loop if at least one filter is active
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

                measurement_pick_pose = None
                measurement_pick_depth = None
                measurement_place_pose = None
                measurement_place_depth = None
                T_base_camera_f = None # Camera pose potentially updated each frame

                # --- Process detected markers in this frame ---
                if ids_f is not None:
                    rvecs_f, tvecs_f, _ = cv2.aruco.estimatePoseSingleMarkers(corners_f, marker_size, camera_matrix, dist_coeffs)
                    cv2.aruco.drawDetectedMarkers(color_image_f, corners_f, ids_f)

                    # Try to update T_base_camera using the base marker if visible
                    base_idx = np.where(ids_f.flatten() == base_marker_id)[0]
                    if len(base_idx) > 0:
                        idx = base_idx[0]
                        T_camera_base_marker = build_transform(rvecs_f[idx][0], tvecs_f[idx][0])
                        T_base_camera_f = T_base_marker_known @ invert_transform(T_camera_base_marker)
                        cv2.drawFrameAxes(color_image_f, camera_matrix, dist_coeffs, rvecs_f[idx][0], tvecs_f[idx][0], marker_size * 0.5)
                    else:
                        T_base_camera_f = T_base_camera # Use locked-in pose if base not seen

                    # Get measurements for PICK marker
                    pick_idx = np.where(ids_f.flatten() == pick_marker_id)[0]
                    if kf_pick and len(pick_idx) > 0 and T_base_camera_f is not None:
                        idx = pick_idx[0]
                        T_camera_pick_marker = build_transform(rvecs_f[idx][0], tvecs_f[idx][0])
                        T_base_pick_marker = T_base_camera_f @ T_camera_pick_marker
                        measurement_pick_pose = T_base_pick_marker[:3, 3]
                        cv2.drawFrameAxes(color_image_f, camera_matrix, dist_coeffs, rvecs_f[idx][0], tvecs_f[idx][0], marker_size * 0.75, thickness=2) # Thicker axes for pick

                        # --- Depth measurement ---
                        marker_corners=corners_f[idx].reshape((4,2));center_x=int(np.mean(marker_corners[:,0]));center_y=int(np.mean(marker_corners[:,1]))
                        if 0<=center_x<intrinsics.width and 0<=center_y<intrinsics.height:
                             depth_value=depth_frame_f.get_distance(center_x,center_y)
                             if depth_value>0.1:
                                 x_cam=(center_x-intrinsics.ppx)*depth_value/intrinsics.fx
                                 y_cam=(center_y-intrinsics.ppy)*depth_value/intrinsics.fy
                                 point_cam=np.array([x_cam,y_cam,depth_value,1.0],dtype=np.float32)
                                 point_base=T_base_camera_f@point_cam
                                 measurement_pick_depth=point_base[:3]

                    # Get measurements for PLACE marker
                    place_idx = np.where(ids_f.flatten() == place_marker_id)[0]
                    if kf_place and len(place_idx) > 0 and T_base_camera_f is not None:
                        idx = place_idx[0]
                        T_camera_place_marker = build_transform(rvecs_f[idx][0], tvecs_f[idx][0])
                        T_base_place_marker = T_base_camera_f @ T_camera_place_marker
                        measurement_place_pose = T_base_place_marker[:3, 3]
                        cv2.drawFrameAxes(color_image_f, camera_matrix, dist_coeffs, rvecs_f[idx][0], tvecs_f[idx][0], marker_size * 0.6, thickness=1) # Thinner axes for place

                        # --- Depth measurement ---
                        marker_corners=corners_f[idx].reshape((4,2));center_x=int(np.mean(marker_corners[:,0]));center_y=int(np.mean(marker_corners[:,1]))
                        if 0<=center_x<intrinsics.width and 0<=center_y<intrinsics.height:
                             depth_value=depth_frame_f.get_distance(center_x,center_y)
                             if depth_value>0.1:
                                 x_cam=(center_x-intrinsics.ppx)*depth_value/intrinsics.fx
                                 y_cam=(center_y-intrinsics.ppy)*depth_value/intrinsics.fy
                                 point_cam=np.array([x_cam,y_cam,depth_value,1.0],dtype=np.float32)
                                 point_base=T_base_camera_f@point_cam
                                 measurement_place_depth=point_base[:3]


                # --- Kalman Filter Update ---
                current_filter_time = time.time()
                dt = current_filter_time - prev_filter_time
                prev_filter_time = current_filter_time

                
                if dt > 0:
                    # Update Pick Filter
                    if kf_pick:
                        kf_pick.predict(dt)
                        if measurement_pick_pose is not None: 
                            kf_pick.update(measurement_pick_pose, kf_pick.R_pose)
                        if measurement_pick_depth is not None:
                            kf_pick.update(measurement_pick_depth, kf_pick.R_depth)
                        # Store potentially updated filtered position
                        filtered_positions_base[pick_marker_id] = kf_pick.x[:3, 0].copy()

                    # Update Place Filter
                    if kf_place:
                        kf_place.predict(dt)
                        if measurement_place_pose is not None: 
                            kf_place.update(measurement_place_pose, kf_place.R_pose)
                        if measurement_place_depth is not None: 
                            kf_place.update(measurement_place_depth, kf_place.R_depth)
                        # Store potentially updated filtered position
                        filtered_positions_base[place_marker_id] = kf_place.x[:3, 0].copy()


                # --- Display Info ---
                # Display Pick Marker Info
                pick_pose_str = f"P{pick_marker_id} Pose: N/A"
                pick_depth_str = f"P{pick_marker_id} Depth: N/A"
                pick_fused_str = f"P{pick_marker_id} Fused: N/A"
                
                if measurement_pick_pose is not None: 
                    pick_pose_str = f"P{pick_marker_id} Pose: {measurement_pick_pose[0]:.3f},{measurement_pick_pose[1]:.3f},{measurement_pick_pose[2]:.3f}"
                if measurement_pick_depth is not None: 
                    pick_depth_str = f"P{pick_marker_id} Depth:{measurement_pick_depth[0]:.3f},{measurement_pick_depth[1]:.3f},{measurement_pick_depth[2]:.3f}"
                if pick_marker_id in filtered_positions_base: 
                    pick_fused_str = f"P{pick_marker_id} Fused:{filtered_positions_base[pick_marker_id][0]:.4f},{filtered_positions_base[pick_marker_id][1]:.4f},{filtered_positions_base[pick_marker_id][2]:.4f}"
                    
                cv2.putText(color_image_f, pick_pose_str, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.putText(color_image_f, pick_depth_str,(10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 0), 1)
                cv2.putText(color_image_f, pick_fused_str,(10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                # Display Place Marker Info
                place_pose_str = f"P{place_marker_id} Pose: N/A"
                place_depth_str = f"P{place_marker_id} Depth: N/A"
                place_fused_str = f"P{place_marker_id} Fused: N/A"
                
                if measurement_place_pose is not None: 
                    place_pose_str = f"P{place_marker_id} Pose: {measurement_place_pose[0]:.3f},{measurement_place_pose[1]:.3f},{measurement_place_pose[2]:.3f}"
                if measurement_place_depth is not None: 
                    place_depth_str = f"P{place_marker_id} Depth:{measurement_place_depth[0]:.3f},{measurement_place_depth[1]:.3f},{measurement_place_depth[2]:.3f}"
                if place_marker_id in filtered_positions_base: 
                    place_fused_str = f"P{place_marker_id} Fused:{filtered_positions_base[place_marker_id][0]:.4f},{filtered_positions_base[place_marker_id][1]:.4f},{filtered_positions_base[place_marker_id][2]:.4f}"
                    
                cv2.putText(color_image_f, place_pose_str, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.putText(color_image_f, place_depth_str,(10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 0), 1)
                cv2.putText(color_image_f, place_fused_str,(10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)


                cv2.imshow("Filter Refinement", color_image_f)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    # If user quits during filtering, return current state
                    print("Filtering aborted by user.")
                    # Return T_base_camera, raw detections, and whatever filtered positions exist
                    return T_base_camera, detected_markers_camera_raw, filtered_positions_base

            # --- End of Filtering Loop ---
            print(f"Filter refinement finished after {frame_count} frames ({time.time() - start_time:.2f}s).")
            if pick_marker_id in filtered_positions_base:
                print(f"Final filtered position for PICK marker {pick_marker_id}: {filtered_positions_base[pick_marker_id].flatten()}")
            else:
                print(f"Warning: Filter refinement did not produce valid position for PICK marker {pick_marker_id}.")
            if place_marker_id in filtered_positions_base:
                 print(f"Final filtered position for PLACE marker {place_marker_id}: {filtered_positions_base[place_marker_id].flatten()}")
            else:
                 print(f"Warning: Filter refinement did not produce valid position for PLACE marker {place_marker_id}.")
        else:
            print("No Kalman filters initialized, skipping refinement phase.")
            # Still return the T_base_camera and raw poses from Phase 1
            # filtered_positions_base will be empty in this case.

    except Exception as e:
        print(f"Error during detection/filtering: {e}")
        import traceback
        traceback.print_exc()
        # Return None for all in case of error
        return None, None, None
    finally:
        if pipeline:
            pipeline.stop()
            print("RealSense pipeline stopped.")
        cv2.destroyAllWindows()
        print("Detection windows closed.")

    # Return camera pose, raw poses, and the dictionary of filtered positions
    return T_base_camera, detected_markers_camera_raw, filtered_positions_base


#######################################
# Motion Execution Functions 
#######################################
# LOGIC: Uses Base Frame Offsets relative to Filtered Position

def calculate_target_pose_in_base(ref_marker_id, base_frame_offset_dict,
                                  target_orientation_base_dict,
                                  T_base_camera, # Still potentially useful for context/debug, but not directly used if filter works
                                  detected_markers_camera_raw, # Still potentially useful for context/debug
                                  filtered_positions_dict=None):
    """
    Calculates target EE pose using filtered marker position and BASE FRAME offsets.
    """
    if not filtered_positions_dict or ref_marker_id not in filtered_positions_dict:
        print(f"Error calculating target: Filtered position for Marker ID {ref_marker_id} not available.")
        return None

    # --- 1. Get Filtered Marker Position ---
    P_marker_base = filtered_positions_dict[ref_marker_id]
    if P_marker_base is None or not np.all(np.isfinite(P_marker_base)):
        print(f"Error calculating target: Filtered position for Marker ID {ref_marker_id} is invalid.")
        return None
    P_marker_base = P_marker_base.flatten() # Ensure it's a 1D array/vector (3,)

    # --- 2. Get Base Frame Offset Vector ---
    V_offset_base = np.array([
        base_frame_offset_dict['x'],
        base_frame_offset_dict['y'],
        base_frame_offset_dict['z']
    ], dtype=np.float32)

    # --- 3. Calculate Target Position via Vector Addition ---
    P_target_base = P_marker_base + V_offset_base
    # print(f"Debug: Marker Pos: {P_marker_base}, Offset: {V_offset_base}, Target Pos: {P_target_base}") # Optional Debug

    # --- 4. Get Target Orientation (Fixed in Base Frame) ---
    R_base_target = eulerAnglesToRotationMatrix([
        target_orientation_base_dict['theta_x'],
        target_orientation_base_dict['theta_y'],
        target_orientation_base_dict['theta_z']
    ])

    # --- 5. Combine Target Position and Orientation ---
    T_base_target = np.eye(4, dtype=np.float32)
    T_base_target[:3, :3] = R_base_target
    T_base_target[:3, 3] = P_target_base # Assign calculated position vector

    return T_base_target

def move_to_pose_relative_to_marker(base, ref_marker_id, position_offset_dict,
                                     target_orientation_base_dict,
                                     T_base_camera, detected_markers_camera_raw,
                                     filtered_positions_dict=None,
                                     blending_radius=0.0):
    print(f"\nAttempting move relative to Marker ID: {ref_marker_id}")
    # --- Pass the filtered positions dictionary ---
    T_base_target = calculate_target_pose_in_base(
        ref_marker_id, position_offset_dict, target_orientation_base_dict,
        T_base_camera, detected_markers_camera_raw, filtered_positions_dict # Pass dict here
    )

    if T_base_target is None:
        print("Move failed: Could not calculate target pose.")
        return False

    # --- Execute Move ---
    try:
        x, y, z, theta_x, theta_y, theta_z = decompose_transform(T_base_target)
        print(f"Target Base Coords: x={x:.4f}, y={y:.4f}, z={z:.4f}, tx={theta_x:.1f}, ty={theta_y:.1f}, tz={theta_z:.1f}")

        waypoint_info = (x, y, z, theta_x, theta_y, theta_z, blending_radius)
        waypoints = Base_pb2.WaypointList()
        waypoints.duration = 0.0 # As fast as possible
        waypoints.use_optimal_blending = False # Keep to avoid potential collisions due to curved paths

        waypoint = waypoints.waypoints.add()
        waypoint.name = f"move_rel_marker_{ref_marker_id}_filt_check" # Waypoint name
        # Populate the cartesian waypoint part
        waypoint.cartesian_waypoint.CopyFrom(populateCartesianCoordinate(waypoint_info))

        # Validate trajectory
        result = base.ValidateWaypointList(waypoints)
        if len(result.trajectory_error_report.trajectory_error_elements) != 0:
            print("FATAL: Trajectory validation failed:")
            print(result.trajectory_error_report)
            return False

        # Execute the trajectory
        e = threading.Event()
        notification_handle = base.OnNotificationActionTopic(
            check_for_end_or_abort(e),
            Base_pb2.NotificationOptions()
        )

        # print("Executing trajectory...")
        base.ExecuteWaypointTrajectory(waypoints)

        # Wait for action completion
        finished = e.wait(TIMEOUT_DURATION)
        base.Unsubscribe(notification_handle)

        print("Move successful." if finished else "Move timed out.")
        return finished
    except Exception as e:
        print(f"Error during move execution: {e}")
        import traceback
        traceback.print_exc()
        return False


# --- WebSocket Callback Functions ---
# Global or passed-in state needed for callbacks
shared_state = {
    "target_key": DEFAULT_PLACE_TARGET_KEY,
    "command": None, # "START_SEQUENCE", "STOP" ,...
    "stop_requested": False, # Flag to signal main loop/threads to stop
    # Add other state info as needed
}
state_lock = threading.Lock() # To protect access to shared_state

def on_message(ws, message):
    global shared_state, state_lock, PLACE_TARGETS # Access shared state/globals
    try:
        data = json.loads(message)
        print(f"WebSocket Received: {data}") # Log received message

        # Get command type and task details from the received data object
        command_type = data.get("type") # "startTask", "stopTask", ...
        task_key = data.get("task")     # "pot1", "pot2", ... (only relevant for startTask)

        if command_type == "startTask":
            if task_key: # Ensure a task key was provided
                with state_lock: # Protect shared state access
                    if task_key in PLACE_TARGETS:
                        print(f"WebSocket: Valid 'startTask' received for target: '{task_key}'")
                        shared_state["target_key"] = task_key  # Update the target
                        shared_state["command"] = "START_SEQUENCE" # Signal main loop to run
                        shared_state["stop_requested"] = False # Clear any previous stop request
                    else:
                        print(f"WebSocket: Received 'startTask' with invalid target key: '{task_key}'")
            else:
                print("WebSocket: Received 'startTask' command without a 'task' field.")

        elif command_type == "stopTask":
            with state_lock: # Protect shared state access
                print("WebSocket: 'stopTask' command received.")
                shared_state["stop_requested"] = True # Signal threads/loops to stop
                shared_state["command"] = "STOP"      # Update command state

        # --- Handling for other potential command types if needed ---
        # elif command_type == "pauseTask":
        #     # (Add logic for pausing if implemented)
        #     print("WebSocket: pauseTask received (not implemented yet).")

        else:
            # Handle messages that aren't commands (e.g., status updates FROM server)
            # Check for 'command_type' == False if server sends other info
            if data.get("command_type") == False:
                 print(f"WebSocket Info: Received non-command message: {data}")
            else:
                 print(f"WebSocket Warning: Received unknown command type: {command_type}")


    except json.JSONDecodeError:
        print(f"WebSocket Error: Received non-JSON message: {message}")
    except Exception as e:
        print(f"WebSocket Error processing message: {e}")
        import traceback
        traceback.print_exc()


def on_error(ws, error):
    print(f"WebSocket Error: {error}")

def on_close(ws, close_status_code, close_msg):
    print(f"### WebSocket Closed: {close_status_code} {close_msg} ###")
    # Attempt reconnection is handled by run_forever(reconnect=...)

def on_open(ws):
    print("WebSocket Opened Connection")
    print("WebSocket: Identifying as robot...")
    # Send identification message using constants from this script
    try:
        ws.send(json.dumps({
            "type": "robot",
            "robot_number": ROBOT_NUMBER,
            "api_key": API_KEY,
            "api_id": API_ID
        }))
        print("WebSocket: Identification sent.")
    except Exception as e:
        print(f"WebSocket Error sending identification: {e}")

# --- WebSocket Handler Function ---
def websocket_handler():
    # === 1) Enable debug trace ===
    websocket.enableTrace(True)

    # === 2) Build and load SSL context ===
    ssl_context = ssl.create_default_context()
    ssl_context.load_verify_locations(CA_PATH)

    # === 3) Create the WebSocketApp ===
    ws = websocket.WebSocketApp(
        WEBSOCKET_URI,
        on_open=on_open,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close
        # header={"Origin": "..."}      # if needed
        # subprotocols=["..."]           # if needed
    )

    # === 4) Run forever, handing control to rel ===
    ws.run_forever(
        sslopt={ 'context': ssl_context },
        dispatcher=rel,
        reconnect=5
    )

    # === 5) Setup clean shutdown and start rel loop ===
    rel.signal(2, rel.abort)   # SIGINT -> abort rel
    rel.signal(15, rel.abort)  # SIGTERM -> abort rel
    rel.dispatch()


def robot_control_thread_worker(args, shared_state, state_lock, shared_status_list):
    print("Starting robot control thread...")
    last_target_key = shared_state.get("target_key") # Init from shared state

    while True:
        time.sleep(0.5) # Check periodically

        # --- Check for Stop Signal ---
        with state_lock:
            should_stop = shared_state["stop_requested"]
            current_command = shared_state.get("command")
            current_target_key = shared_state["target_key"]

        if should_stop:
            print("Robot control thread: Stop requested. Exiting.")
            break # Exit the loop

        # --- Check for Start Command / Target Change (Adapt logic as needed) ---
        run_now = False
        if current_command == "START_SEQUENCE":
             print(f"Robot control thread: START command received for target '{current_target_key}'.")
             run_now = True
             with state_lock: # Reset command
                 shared_state["command"] = None
        # Or trigger on target change:
        # elif current_target_key != last_target_key:
        #     print(f"Robot control thread: Target changed to '{current_target_key}'. Triggering run.")
        #     run_now = True
        #     last_target_key = current_target_key


        # --- Execute Sequence if Triggered ---
        if run_now:
            if current_target_key not in PLACE_TARGETS:
                print(f"Robot control thread: Invalid target key '{current_target_key}'.")
                shared_status_list[0] = f"Idle - Invalid Target: {current_target_key}"
                continue

            print(f"\n===== Robot Control: Starting Sequence for Target: {current_target_key} =====")
            shared_status_list[0] = f"Starting Sequence: {current_target_key}"
            sequence_result = False
            try:
                # Establish connection for this sequence run
                print("Robot control thread: Establishing connection...")
                with utilities.DeviceConnection.createTcpConnection(args) as router:
                    print("Robot control thread: Connection established.")
                    base = BaseClient(router)
                    sequence_result = run_pick_place_return_sequence(
                        base, current_target_key, shared_status_list, state_lock
                    )
            except Exception as conn_err:
                 print(f"Robot control thread: Connection/Sequence Error: {conn_err}")
                 shared_status_list[0] = "Error - Connection/Sequence Failed"
                 import traceback
                 traceback.print_exc()

            print(f"===== Robot Control: Sequence Finished (Success: {sequence_result}) =====")
            shared_status_list[0] = f"Idle - Last Run Success: {sequence_result}"

    print("Robot control thread finished.")


# --- Robot Task Sequence ---
def run_pick_place_return_sequence(base, target_key, shared_status_list, state_lock):
    """
    Performs the entire robot motion sequence for a given target key.
    Returns True on success, False on failure.
    Updates shared_status_list.
    Checks shared_state["stop_requested"] periodically.
    """
    global shared_state # Need access to check stop_requested flag
    sequence_success = True
    current_local_status = "Starting sequence"

    try:
        # --- Get Target Info ---
        if target_key not in PLACE_TARGETS:
            print(f"Error: Invalid target key '{target_key}' provided to sequence.")
            return False
        place_info = PLACE_TARGETS[target_key]
        local_place_marker_id = place_info["marker_id"]
        local_place_approach_offset = place_info["approach_offset"] # Assumes base frame offsets now
        local_place_hold_offset = place_info["hold_offset"]         # Assumes base frame offsets now

        # --- CHECK FOR STOP REQUEST ---
        with state_lock:
            if shared_state["stop_requested"]: raise InterruptedError("Stop requested before start")

        # --- 1. Move Home ---
        current_local_status = "Sequence: Moving Home"
        shared_status_list[0] = current_local_status
        if not example_move_to_home_position(base): raise RuntimeError("Failed initial home")

        with state_lock:
             if shared_state["stop_requested"]: raise InterruptedError("Stop requested after home")

        # --- 2. Open Gripper ---
        current_local_status = "Sequence: Opening Gripper"
        shared_status_list[0] = current_local_status
        open_gripper(base)

        # --- 3. Detect & Filter ---
        current_local_status = "Sequence: Detecting/Filtering"
        shared_status_list[0] = current_local_status
        print("\n--- Sequence: Starting Detection & Filtering ---")
        required_markers = [PICK_MARKER_ID, local_place_marker_id]
        if BASE_MARKER_ID not in required_markers: required_markers.append(BASE_MARKER_ID)

        local_T_base_camera, local_detected_raw, local_filtered_pos = detect_markers_and_filter(
            MARKER_SIZE, required_markers, BASE_MARKER_ID, PICK_MARKER_ID,
            local_place_marker_id, T_base_marker, FILTER_DURATION
        )

        # Validate results
        if local_T_base_camera is None or local_detected_raw is None or local_filtered_pos is None:
            raise RuntimeError("Detection/Filtering phase failed.")
        if PICK_MARKER_ID not in local_filtered_pos:
            raise RuntimeError(f"Failed filtered pos for PICK marker {PICK_MARKER_ID}.")
        if local_place_marker_id not in local_filtered_pos:
            raise RuntimeError(f"Failed filtered pos for PLACE marker {local_place_marker_id}.")
        print(f"--- Sequence: Detection Complete. Filtered: {local_filtered_pos} ---")

        with state_lock:
             if shared_state["stop_requested"]: raise InterruptedError("Stop requested after detect")

        # --- 4-7. Pick Sequence ---
        print("\n--- Sequence: Starting Pick ---")
        current_local_status = "Sequence: Pick Approach"
        shared_status_list[0] = current_local_status
        if not move_to_pose_relative_to_marker(base, PICK_MARKER_ID, PICK_APPROACH_POS_OFFSET, TARGET_ORIENTATION_BASE, local_T_base_camera, local_detected_raw, local_filtered_pos): 
            raise RuntimeError("Pick approach failed")

        with state_lock:
             if shared_state["stop_requested"]: 
                 raise InterruptedError("Stop requested during pick")

        current_local_status = "Sequence: Pick Descend"
        shared_status_list[0] = current_local_status
        if not move_to_pose_relative_to_marker(base, PICK_MARKER_ID, PICK_GRASP_POS_OFFSET, TARGET_ORIENTATION_BASE, local_T_base_camera, local_detected_raw, local_filtered_pos): 
            raise RuntimeError("Pick descend failed")

        current_local_status = "Sequence: Pick Grasping"
        shared_status_list[0] = current_local_status
        close_gripper(base)
        time.sleep(0.5)

        current_local_status = "Sequence: Pick Ascend"
        shared_status_list[0] = current_local_status
        if not move_to_pose_relative_to_marker(base, PICK_MARKER_ID, PICK_LIFT_POS_OFFSET, TARGET_ORIENTATION_BASE, local_T_base_camera, local_detected_raw, local_filtered_pos): 
            raise RuntimeError("Pick ascend failed")
        print("--- Sequence: Pick Complete ---")

        with state_lock:
             if shared_state["stop_requested"]: 
                 raise InterruptedError("Stop requested after pick")

        # --- 8. Home (Holding) ---
        current_local_status = "Sequence: Homing (Holding)"
        shared_status_list[0] = current_local_status
        print("\n--- Sequence: Homing (Object Held) ---")
        if not example_move_to_home_position(base): 
            raise RuntimeError("Homing after pick failed")
        time.sleep(1.0)

        # --- 9-12. Place Sequence ---
        print(f"\n--- Sequence: Starting Place (Target: {target_key}) ---")
        current_local_status = f"Sequence: Place Approach M:{local_place_marker_id}"
        shared_status_list[0] = current_local_status
        if not move_to_pose_relative_to_marker(base, local_place_marker_id, local_place_approach_offset, TARGET_ORIENTATION_BASE, local_T_base_camera, local_detected_raw, local_filtered_pos): 
            raise RuntimeError("Place approach failed")

        with state_lock:
             if shared_state["stop_requested"]: 
                 raise InterruptedError("Stop requested during place")

        current_local_status = f"Sequence: Place Descend M:{local_place_marker_id}"
        shared_status_list[0] = current_local_status
        if not move_to_pose_relative_to_marker(base, local_place_marker_id, local_place_hold_offset, TARGET_ORIENTATION_BASE, local_T_base_camera, local_detected_raw, local_filtered_pos): 
            raise RuntimeError("Place descend failed")

        current_local_status = f"Sequence: Place Holding M:{local_place_marker_id}"
        shared_status_list[0] = current_local_status
        print(f"Holding at {target_key} for {HOLD_DURATION:.1f}s...")
        # Check for stop signal periodically during wait
        wait_start = time.time()
        while time.time() - wait_start < HOLD_DURATION:
            with state_lock:
                if shared_state["stop_requested"]: raise InterruptedError("Stop requested during hold")
            time.sleep(0.1) # Sleep briefly
        print("Hold finished.")

        current_local_status = f"Sequence: Place Ascend M:{local_place_marker_id}"
        shared_status_list[0] = current_local_status
        if not move_to_pose_relative_to_marker(base, local_place_marker_id, local_place_approach_offset, TARGET_ORIENTATION_BASE, local_T_base_camera, local_detected_raw, local_filtered_pos): 
            raise RuntimeError("Place ascend failed")
        print("--- Sequence: Place Complete ---")

        with state_lock:
             if shared_state["stop_requested"]: raise InterruptedError("Stop requested after place")

        # --- 13. Home (Holding) ---
        current_local_status = "Sequence: Homing (Holding)"
        shared_status_list[0] = current_local_status
        print("\n--- Sequence: Homing (Object Held) ---")
        if not example_move_to_home_position(base): 
            raise RuntimeError("Homing after place failed")

        # --- 14-16. Return Sequence ---
        print("\n--- Sequence: Starting Return ---")
        current_local_status = "Sequence: Return to Original Pos"
        shared_status_list[0] = current_local_status
        if not move_to_pose_relative_to_marker(base, PICK_MARKER_ID, PICK_LIFT_POS_OFFSET, TARGET_ORIENTATION_BASE, local_T_base_camera, local_detected_raw, local_filtered_pos): 
            raise RuntimeError("Return move failed")

        with state_lock:
             if shared_state["stop_requested"]: 
                 raise InterruptedError("Stop requested during return")

        current_local_status = "Sequence: Return Descending"
        shared_status_list[0] = current_local_status
        if not move_to_pose_relative_to_marker(base, PICK_MARKER_ID, PICK_GRASP_POS_OFFSET, TARGET_ORIENTATION_BASE, local_T_base_camera, local_detected_raw, local_filtered_pos): 
            raise RuntimeError("Return ascend failed")
        print("--- Sequence: Return Complete ---")
        
        with state_lock:
             if shared_state["stop_requested"]: 
                 raise InterruptedError("Stop requested during return")

        current_local_status = "Sequence: Return Releasing"
        shared_status_list[0] = current_local_status
        print("Releasing object...")
        open_gripper(base)
        print("Object released.")

        current_local_status = "Sequence: Return Ascending"
        shared_status_list[0] = current_local_status
        if not move_to_pose_relative_to_marker(base, PICK_MARKER_ID, PICK_APPROACH_POS_OFFSET, TARGET_ORIENTATION_BASE, local_T_base_camera, local_detected_raw, local_filtered_pos): 
            raise RuntimeError("Return ascend failed")
        print("--- Sequence: Return Complete ---")

        # --- 17. Final Home ---
        current_local_status = "Sequence: Final Homing"
        shared_status_list[0] = current_local_status
        print("\n--- Sequence: Final Homing ---")
        if not example_move_to_home_position(base):
             print("Warning: Final home failed.")
             sequence_success = False # Mark as failure if final home fails
        else:
             print("--- Sequence: Complete & Homed ---")

    except InterruptedError as ie:
        print(f"SEQUENCE INTERRUPTED: {ie}")
        shared_status_list[0] = f"Sequence Interrupted: {current_local_status}"
        sequence_success = False # Mark as unsuccessful if interrupted
        # Attempt to move home safely after interruption
        try:
             if base:
                  print("Attempting to home after interruption...")
                  # Might need to ensure gripper is open first depending on state
                  # open_gripper(base) # If needed, avoided to keep object from dropping
                  example_move_to_home_position(base)
        except Exception as home_e:
             print(f"Error during homing after interruption: {home_e}")

    except Exception as e:
        print(f"\n!!! ERROR during sequence execution: {e} !!!")
        shared_status_list[0] = f"Error during: {current_local_status}"
        import traceback
        traceback.print_exc()
        sequence_success = False
        # Attempt to move home safely after error
        try:
             if base:
                  print("Attempting to home after error...")
                  example_move_to_home_position(base)
        except Exception as home_e:
             print(f"Error during homing after error: {home_e}")

    return sequence_success



# --- Main Function (Runs WS in Main Thread) ---
def main():
    global shared_state, state_lock # Allow access

    # --- Initialization ---
    start_time = time.time()
    shared_status_list = ["Initializing"]
    stop_reporting_event = threading.Event()
    status_thread = None
    robot_thread = None # Thread for robot control loop

    # --- Outer Try/Except/Finally ---
    try:
        # --- Argument Parsing ---
        try: import utilities
        except ImportError: print("ERROR: Failed to import 'utilities'."); return 1
        except NameError: import utilities
        args = utilities.parseConnectionArguments()
        if args is None: return 1

        # --- Start Status Reporting Thread ---
        print("Starting status reporting thread...")
        status_thread = threading.Thread(
            target=status_reporting_worker,
            args=(stop_reporting_event, shared_status_list, start_time),
            daemon=True
        )
        status_thread.start()

        # --- Start Robot Control Thread ---
        print("Starting robot control thread...")
        robot_thread = threading.Thread(
            target=robot_control_thread_worker,
            args=(args, shared_state, state_lock, shared_status_list),
            daemon=True
        )
        robot_thread.start()

        # --- Setup Signal Handling for Main Thread (WebSocket/rel loop) ---
        # Handle Ctrl+C gracefully to stop the WebSocket loop and signal threads
        print("Setting up signal handlers for main thread...")
        try:
             # Capture SIGINT (Ctrl+C) and SIGTERM
             rel.signal(2, rel.abort) # SIGINT -> call rel.abort()
             rel.signal(15, rel.abort) # SIGTERM -> call rel.abort()
             print("Signal handlers set using rel.")
        except ValueError as e:
             # This can happen on Windows which has limited signal support
             print(f"Warning: Could not set rel signal handlers ({e}). Use Ctrl+Break or task manager.")


        # --- Run WebSocket Handler in Main Thread ---
        # This call will block until the WebSocket loop finishes or is aborted
        websocket_handler() # Runs ws.run_forever(dispatcher=rel)

        # --- Code here executes after websocket_handler finishes ---
        print("WebSocket handler has exited. Requesting stop for other threads...")
        with state_lock:
            shared_state["stop_requested"] = True # Ensure flag is set

    except KeyboardInterrupt:
         print("\nCtrl+C detected in main thread. Requesting stop...")
         with state_lock:
             shared_state["stop_requested"] = True
         try:
             rel.abort() # Try to stop the event loop if running
         except Exception as rel_e:
              print(f"Error aborting rel on KeyboardInterrupt: {rel_e}")

    except Exception as e:
        print(f"\n!!! Critical error in main execution: {e} !!!")
        import traceback
        traceback.print_exc()
        with state_lock: # Signal stop on critical error
            shared_state["stop_requested"] = True
        return 1

    finally:
        # --- Cleanup ---
        print("Main function exiting. Cleaning up threads...")
        # Signal status reporting thread
        stop_reporting_event.set()

        # Threads were signaled via shared_state["stop_requested"] = True

        # Wait for threads
        if status_thread and status_thread.is_alive():
            print("Waiting for status reporting thread...")
            status_thread.join(timeout=2.0)
            if status_thread.is_alive(): print("Warning: Status thread did not stop.")

        if robot_thread and robot_thread.is_alive():
            print("Waiting for robot control thread...")
            robot_thread.join(timeout=5.0) # Give robot thread more time
            if robot_thread.is_alive(): print("Warning: Robot control thread did not stop.")

        print("Cleanup complete. Exiting.")

    return 0


# --- Script execution block ---
if __name__ == "__main__":
    main_exit_code = main()
    # Ensure rel event loop is stopped if main exits prematurely
    # This might already be handled by rel.abort() calls
    try:
         rel.abort()
    except: # Ignore errors if rel wasn't running/setup
         pass
    print(f"Exiting with code: {main_exit_code}")
    sys.exit(main_exit_code)
