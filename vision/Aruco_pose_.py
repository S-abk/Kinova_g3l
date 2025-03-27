import pyrealsense2 as rs
import cv2
import numpy as np

# --- Setup RealSense Pipeline ---
pipeline = rs.pipeline()
config = rs.config()
# Configure streams for both depth and color (adjust resolution as needed)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming and get the profile
profile = pipeline.start(config)

# --- Retrieve Camera Intrinsics ---
# Get the video stream profile and extract intrinsics
color_stream = profile.get_stream(rs.stream.color)
color_profile = color_stream.as_video_stream_profile()
intrinsics = color_profile.get_intrinsics()

# Create the camera matrix and distortion coefficients
camera_matrix = np.array([[intrinsics.fx, 0, intrinsics.ppx],
                           [0, intrinsics.fy, intrinsics.ppy],
                           [0, 0, 1]])
dist_coeffs = np.array(intrinsics.coeffs)
print("Camera matrix:\n", camera_matrix)
print("Distortion coefficients:\n", dist_coeffs)

# --- Setup ArUco Detector ---
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
aruco_params = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

# Define the physical size of your marker (in meters)
marker_length = 0.095  #9.5 cm marker

try:
    while True:
        # Wait for a new frame from the RealSense camera
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        # Convert the color frame to a numpy array
        color_image = np.asanyarray(color_frame.get_data())
        # Convert to grayscale for marker detection
        gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

        # Detect ArUco markers in the image
        corners, ids, rejected = detector.detectMarkers(gray)

        if ids is not None:
            # Draw detected markers for visualization
            cv2.aruco.drawDetectedMarkers(color_image, corners, ids)
            
            # Estimate the pose of each marker
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners,
                                                                   marker_length,
                                                                   camera_matrix,
                                                                   dist_coeffs)
            for rvec, tvec in zip(rvecs, tvecs):
                # Draw the axis for each marker (axis length set to half the marker size)
                cv2.drawFrameAxes(color_image, camera_matrix, dist_coeffs, rvec, tvec, marker_length * 0.5)
                # print out the pose information
                print("Rotation Vector:\n", rvec)
                print("Translation Vector:\n", tvec)

        # Display the result with the drawn markers and pose axes
        cv2.imshow("Pose Estimation", color_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Cleanup: stop the pipeline and close windows
    pipeline.stop()
    cv2.destroyAllWindows()
