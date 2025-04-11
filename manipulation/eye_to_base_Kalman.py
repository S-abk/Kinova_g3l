import pyrealsense2 as rs
import numpy as np
import cv2
import cv2.aruco as aruco
import time

# -----------------------------
# 1. Helper Functions for Transforms
# -----------------------------
def build_transform(rvec, tvec):
    """
    Build a 4x4 homogeneous transformation matrix from rvec and tvec.
    """
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = R
    T[:3, 3] = tvec.reshape(3)
    return T

def invert_transform(T):
    """
    Invert a 4x4 homogeneous transformation matrix.
    """
    R = T[:3, :3]
    t = T[:3, 3]
    R_inv = R.T
    t_inv = -R_inv @ t
    T_inv = np.eye(4, dtype=np.float32)
    T_inv[:3, :3] = R_inv
    T_inv[:3, 3] = t_inv
    return T_inv

# -----------------------------
# 2. Kalman Filter for 6D State (3D position + 3D velocity)
# -----------------------------
class KalmanFilter6D:
    def __init__(self, dt, initial_state):
        # State vector: [x, y, z, vx, vy, vz]^T
        self.x = initial_state  # shape (6,1)
        # Initial state covariance
        self.P = np.eye(6) * 1.0
        
        # State transition matrix for constant velocity model
        self.A = np.eye(6)
        self.A[0, 3] = dt
        self.A[1, 4] = dt
        self.A[2, 5] = dt
        
        # Process noise covariance (tune these values as needed)
        self.Q = np.diag([0.01, 0.01, 0.01, 0.1, 0.1, 0.1])
        
        # Measurement matrix: we measure position only
        self.H = np.hstack((np.eye(3), np.zeros((3, 3))))
        
        # Measurement noise covariances for the two sensor types
        self.R_pose = np.eye(3) * 0.05   # for pose estimation measurement
        self.R_depth = np.eye(3) * 0.1     # for direct depth (back-projected) measurement

    def predict(self, dt):
        # Update state transition matrix with current dt
        self.A[0, 3] = dt
        self.A[1, 4] = dt
        self.A[2, 5] = dt
        
        # Predict the state and covariance
        self.x = self.A @ self.x
        self.P = self.A @ self.P @ self.A.T + self.Q

    def update(self, z, R):
        """
        Update the filter with measurement z (3x1 vector) and measurement noise covariance R.
        """
        z = z.reshape((3, 1))
        y = z - self.H @ self.x                   # measurement residual
        S = self.H @ self.P @ self.H.T + R          # residual covariance
        K = self.P @ self.H.T @ np.linalg.inv(S)    # Kalman gain
        self.x = self.x + K @ y
        self.P = (np.eye(6) - K @ self.H) @ self.P

# -----------------------------
# 3. RealSense, ArUco, and Calibration Setup
# -----------------------------
# Initialize RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
profile = pipeline.start(config)

# Align depth to color
align_to = rs.stream.color
align = rs.align(align_to)

# Get camera intrinsics
color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
intrinsics = color_profile.get_intrinsics()
camera_matrix = np.array([[intrinsics.fx, 0, intrinsics.ppx],
                           [0, intrinsics.fy, intrinsics.ppy],
                           [0, 0, 1]])
dist_coeffs = np.array(intrinsics.coeffs)

# Setup ArUco dictionary and parameters
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
aruco_params = cv2.aruco.DetectorParameters()

# Marker parameters
marker_length = 0.04  # marker size in meters
base_marker_id = 8    # known base marker
object_marker_id = 3  # object marker for manipulation

# Known transform from manipulator base to base marker frame
T_base_marker = np.array([
    [1, 0, 0, -0.09],
    [0, 1, 0,  0.0 ],
    [0, 0, 1,  0.0 ],
    [0, 0, 0,  1.0 ]
], dtype=np.float32)

# -----------------------------
# 4. Initialize the Kalman Filter
# -----------------------------
# We'll initialize the filter with a guessed position (e.g., [0,0,1] in meters) and zero velocity.
initial_state = np.array([[0.0], [0.0], [1.0], [0.0], [0.0], [0.0]])
dt_initial = 1.0 / 30.0  # initial time step based on 30 FPS
kf = KalmanFilter6D(dt_initial, initial_state)

prev_time = time.time()

# -----------------------------
# 5. Main Loop: Detection, Measurements, and Fusion
# -----------------------------
try:
    while True:
        # Acquire frames
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        # Convert to numpy arrays and grayscale
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

        # Detect markers
        corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=aruco_params)
        T_base_camera = None  # Transformation from camera frame to manipulator base frame

        # -----------------------------
        # 5.1 Compute T_base_camera from the Base Marker (ID 8)
        # -----------------------------
        if ids is not None:
            for i, marker_id in enumerate(ids.flatten()):
                if marker_id == base_marker_id:
                    # Estimate pose for the base marker
                    rvec_base, tvec_base, _ = cv2.aruco.estimatePoseSingleMarkers(
                        [corners[i]], marker_length, camera_matrix, dist_coeffs
                    )
                    rvec_base = rvec_base[0][0]
                    tvec_base = tvec_base[0][0]
                    T_camera_marker = build_transform(rvec_base, tvec_base)
                    T_marker_camera = invert_transform(T_camera_marker)
                    T_base_camera = T_base_marker @ T_marker_camera
                    # Draw axes for visualization
                    cv2.drawFrameAxes(color_image, camera_matrix, dist_coeffs, rvec_base, tvec_base, marker_length * 0.5)
                    break

        # Initialize measurement placeholders
        measurement_pose = None   # 3D position from pose estimation (object marker)
        measurement_depth = None  # 3D position from direct depth measurement

        # -----------------------------
        # 5.2 Compute Object Marker Pose (ID 3) and the Direct Depth Measurement
        # -----------------------------
        if T_base_camera is not None and ids is not None:
            for i, marker_id in enumerate(ids.flatten()):
                if marker_id == object_marker_id:
                    # Estimate pose for the object marker
                    rvec_obj, tvec_obj, _ = cv2.aruco.estimatePoseSingleMarkers(
                        [corners[i]], marker_length, camera_matrix, dist_coeffs
                    )
                    rvec_obj = rvec_obj[0][0]
                    tvec_obj = tvec_obj[0][0]
                    T_camera_obj = build_transform(rvec_obj, tvec_obj)
                    # Transform object marker pose to manipulator base frame
                    T_base_obj = T_base_camera @ T_camera_obj
                    measurement_pose = T_base_obj[:3, 3]  # 3D position

                    # Also, get the direct depth measurement:
                    # Use the marker's center pixel to back-project to 3D in the camera frame.
                    marker_corners = corners[i].reshape((4, 2))
                    center_x = int(np.mean(marker_corners[:, 0]))
                    center_y = int(np.mean(marker_corners[:, 1]))
                    depth_value = depth_frame.get_distance(center_x, center_y)
                    if depth_value > 0:
                        # Back-project to 3D (in camera coordinates)
                        x_cam = (center_x - intrinsics.ppx) * depth_value / intrinsics.fx
                        y_cam = (center_y - intrinsics.ppy) * depth_value / intrinsics.fy
                        point_cam = np.array([x_cam, y_cam, depth_value, 1.0], dtype=np.float32)
                        # Transform to base frame using T_base_camera
                        point_base = T_base_camera @ point_cam
                        measurement_depth = point_base[:3]

                    # Draw marker and axes for visualization
                    cv2.aruco.drawDetectedMarkers(color_image, corners, ids)
                    cv2.drawFrameAxes(color_image, camera_matrix, dist_coeffs, rvec_obj, tvec_obj, marker_length * 0.5)
                    # Optionally, annotate the image with the measurements
                    cv2.putText(color_image, f"Pose: {measurement_pose[0]:.3f}, {measurement_pose[1]:.3f}, {measurement_pose[2]:.3f}",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    if measurement_depth is not None:
                        cv2.putText(color_image, f"Depth: {measurement_depth[0]:.3f}, {measurement_depth[1]:.3f}, {measurement_depth[2]:.3f}",
                                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                    break

        # -----------------------------
        # 5.3 Kalman Filter Prediction and Measurement Updates
        # -----------------------------
        current_time = time.time()
        dt = current_time - prev_time
        prev_time = current_time
        
        kf.predict(dt)
        
        # Update with the full pose measurement if available
        if measurement_pose is not None:
            kf.update(measurement_pose, kf.R_pose)
        # Update with the back-projected depth measurement if available
        if measurement_depth is not None:
            kf.update(measurement_depth, kf.R_depth)
        
        fused_position = kf.x[:3, 0]  # Extract the 3D position
        
        # Display the fused position on the image
        cv2.putText(color_image, f"Fused Pos: {fused_position[0]:.3f}, {fused_position[1]:.3f}, {fused_position[2]:.3f}",
                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # -----------------------------
        # 5.4 Display the Results
        # -----------------------------
        cv2.imshow("Kalman Filter Pose Fusion", color_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
