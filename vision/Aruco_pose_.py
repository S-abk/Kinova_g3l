import cv2
import numpy as np
import requests
from io import BytesIO
from PIL import Image

image_url = "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/left01.jpg"
response = requests.get(image_url)
image = np.array(Image.open(BytesIO(response.content)))

if image is None:
    raise ValueError("Image not loaded. Please check the image path or URL.")

# Check the number of channels in the image
if len(image.shape) == 2:  # Grayscale image
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
elif image.shape[2] == 4:  # RGBA image
    image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Define the dictionary and parameters
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
parameters = cv2.aruco.DetectorParameters()

# Create the ArUco detector
detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

# Detect the markers
corners, ids, rejected = detector.detectMarkers(gray)
print("Detected markers:", ids)

if ids is not None:
    cv2.aruco.drawDetectedMarkers(image, corners, ids)
    cv2.imshow('Detected Markers', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
