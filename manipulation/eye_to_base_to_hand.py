import pyrealsense2 as rs
import cv2
import numpy as np

# -----------------------------
# 1. Setup RealSense Pipeline
# -----------------------------
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
profile = pipeline.start(config)

# Retrieve camera intrinsics (for pose estimation)
color_stream = profile.get_stream(rs.stream.color)
color_profile = color_stream.as_video_stream_profile()
intrinsics = color_profile.get_intrinsics()
camera_matrix = np.array([[intrinsics.fx, 0, intrinsics.ppx],
                           [0, intrinsics.fy, intrinsics.ppy],
                           [0, 0, 1]])
dist_coeffs = np.array(intrinsics.coeffs)

print("Camera matrix:\n", camera_matrix)
print("Distortion coefficients:\n", dist_coeffs)

# -----------------------------
# 2. Setup ArUco Detector
# -----------------------------
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
aruco_params = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

# Marker parameters
marker_length = 0.04  # 40 mm markers (0.04 m)
base_marker_id = 8    # Known base marker
object_marker_id = 3  # Object marker to pick

# Known transform for the base marker in the manipulator base frame.
# Its origin is offset by -0.07 m in the X direction, with no rotation.
T_base_marker = np.array([
    [1, 0, 0, -0.07],
    [0, 1, 0,  0.0 ],
    [0, 0, 1,  0.0 ],
    [0, 0, 0,  1.0 ]
], dtype=np.float32)

# -----------------------------
# 3. Helper Functions
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
# 4. Main Loop: Detection & Transformation
# -----------------------------
try:
    while True:
        # Get the next set of frames from the camera
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        # Convert the image to a numpy array and grayscale
        color_image = np.asanyarray(color_frame.get_data())
        gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

        # Detect ArUco markers in the image
        corners, ids, _ = detector.detectMarkers(gray)
        if ids is not None:
            cv2.aruco.drawDetectedMarkers(color_image, corners, ids)

        T_base_camera = None  # Will hold the transformation from manipulator base to camera

        # -----------------------------
        # 4.1. Find the Base Marker (ID 8)
        # -----------------------------
        if ids is not None:
            for i, marker_id in enumerate(ids.flatten()):
                if marker_id == base_marker_id:
                    # Estimate pose for the base marker
                    rvec_base, tvec_base, _ = cv2.aruco.estimatePoseSingleMarkers(
                        [corners[i]], marker_length, camera_matrix, dist_coeffs
                    )
                    # Extract the single result (arrays have shape (1,1,3))
                    rvec_base = rvec_base[0][0]
                    tvec_base = tvec_base[0][0]

                    # Build the transform from camera to marker frame
                    T_camera_marker = build_transform(rvec_base, tvec_base)
                    # Invert it to get marker-to-camera transform
                    T_marker_camera = invert_transform(T_camera_marker)
                    # Now compute the transform from base to camera:
                    # T_base_camera = T_base_marker * T_marker_camera
                    T_base_camera = T_base_marker @ T_marker_camera

                    # Draw coordinate axes for visualization
                    cv2.drawFrameAxes(color_image, camera_matrix, dist_coeffs,
                                      rvec_base, tvec_base, marker_length * 0.5)
                    break

        # -----------------------------
        # 4.2. Find the Object Marker (ID 3) and Transform Its Pose
        # -----------------------------
        if T_base_camera is not None and ids is not None:
            for i, marker_id in enumerate(ids.flatten()):
                if marker_id == object_marker_id:
                    rvec_obj, tvec_obj, _ = cv2.aruco.estimatePoseSingleMarkers(
                        [corners[i]], marker_length, camera_matrix, dist_coeffs
                    )
                    rvec_obj = rvec_obj[0][0]
                    tvec_obj = tvec_obj[0][0]
                    # Build transform from camera to object marker
                    T_camera_obj = build_transform(rvec_obj, tvec_obj)
                    # Transform to manipulator base frame:
                    T_base_obj = T_base_camera @ T_camera_obj
                    # Extract the translation (position)
                    pos_base = T_base_obj[:3, 3]
                    
                    # Draw coordinate axes on the object marker
                    cv2.drawFrameAxes(color_image, camera_matrix, dist_coeffs,
                                      rvec_obj, tvec_obj, marker_length * 0.5)
                    # Display the position on the image
                    cv2.putText(color_image,
                                f"Obj Pos: {pos_base[0]:.3f}, {pos_base[1]:.3f}, {pos_base[2]:.3f}",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    print("Object marker (ID 3) position in base frame:", pos_base)
                    break

        # Show the result image
        cv2.imshow("ArUco Detection", color_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
