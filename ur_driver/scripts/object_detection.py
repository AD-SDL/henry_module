# import pyrealsense2 as rs
# import numpy as np

# pipeline = rs.pipeline()

# # Start streaming
# pipeline.start()

# frames = pipeline.wait_for_frames()
# depth = frames.get_depth_frame()

# width = depth.get_width()
# height = depth.get_height()

# dist = depth.get_distance(int(width/2), int(height/2))
# print(dist)

# # Stop streaming
# pipeline.stop()

# # Source of code: https://github.com/IntelRealSense/librealsense/issues/9939



import cv2 as cv                               # state of the art computer vision algorithms library
import numpy as np                        # fundamental package for scientific computing
import matplotlib.pyplot as plt           # 2D plotting library producing publication quality figures
import pyrealsense2 as rs 


# Create a pipeline to continuous collect data from intel camera
pipe = rs.pipeline()

# Configure the pipeline
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)  # Color stream configuration
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)  # Depth stream configuration

# Start the pipeline
profile = pipe.start(config)


while True:
    # Wait for the next set of frames
    frames = pipe.wait_for_frames()

    # Get color and depth frames
    color_frame = frames.get_color_frame()
    depth_frame = frames.get_depth_frame()

    if not color_frame or not depth_frame:
        continue

    # Convert the frames to numpy arrays
    color_image = np.asanyarray(color_frame.get_data())
    depth_image = np.asanyarray(depth_frame.get_data())

    # Convert the depth image to meters
    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
    depth_image = depth_image * depth_scale

    # Obtain the depth value at a specific pixel
    x, y = 320, 240  # Example pixel coordinates, adjust as needed
    depth_value = depth_image[y, x]


    # Calculate the distance using the depth value
    distance = depth_value

    # Print the distance
    print("Distance to object at pixel ({}, {}): {:.3f} meters".format(x, y, distance))

    # Display the frames, use the camera
    cv.imshow('Color', color_image)
    cv.imshow('Depth', depth_image)

    # Exit the loop if 'q' is pressed
    if cv.waitKey(1) & 0xFF == ord('q'):
        break


# Stop the pipeline and release resources
pipe.stop()
cv.destroyAllWindows()

#example of detecting image using CascadeClassifier and detectMultiScale
# import cv2

# # Load the pre-trained Haar cascade classifier
# face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# # Load the image
# image = cv2.imread('image.jpg')

# # Convert the image to grayscale
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# # Detect faces in the image
# faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

# # Draw bounding boxes around the detected faces
# for (x, y, w, h) in faces:
#     cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

# # Display the image with detected faces
# cv2.imshow('Faces Detected', image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

