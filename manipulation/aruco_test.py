import pyrealsense2 as rs
import numpy as np
import cv2

# --- Setup RealSense Pipeline ---
pipeline = rs.pipeline()
config = rs.config()

# Configure the pipeline to stream both color and depth
# Adjust resolution and framerate as needed for your device
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
profile = pipeline.start(config)

# Optionally, align depth to color
align_to = rs.stream.color
align = rs.align(align_to)

# --- Load ArUco Dictionary and Parameters ---
aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
aruco_params = cv2.aruco.DetectorParameters_create()

try:
    while True:
        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        # Align the depth frame to color frame
        aligned_frames = align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # --- ArUco Detection ---
        # Convert to grayscale as required for marker detection
        gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=aruco_params)

        if ids is not None:
            # Draw detected markers on the color image
            cv2.aruco.drawDetectedMarkers(color_image, corners, ids)
            
            # Optionally, get the depth for each marker (using the center point)
            for marker_corners, marker_id in zip(corners, ids):
                # Calculate the center of the marker
                marker_corners = marker_corners.reshape((4, 2))
                center_x = int(np.mean(marker_corners[:, 0]))
                center_y = int(np.mean(marker_corners[:, 1]))
                
                # Get the depth at the center (in millimeters)
                depth = depth_frame.get_distance(center_x, center_y)
                
                # Display marker ID and depth on the image
                cv2.putText(color_image, f'ID:{marker_id[0]} Depth:{depth:.2f}m',
                            (center_x - 50, center_y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Display the color image with detected markers
        cv2.imshow('ArUco Marker Detection', color_image)
        
        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Stop streaming
    pipeline.stop()
    cv2.destroyAllWindows()
