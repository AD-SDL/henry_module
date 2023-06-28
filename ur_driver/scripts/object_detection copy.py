# import cv2 as cv                               # state of the art computer vision algorithms library
# import numpy as np                        # fundamental package for scientific computing
# import matplotlib.pyplot as plt           # 2D plotting library producing publication quality figures
# import pyrealsense2 as rs 
# import time as t

# # Create a pipeline to start streaming on camera
# pipe = rs.pipeline()

# # Configuration prepares the camera to be set up for recording
# config = rs.config()
# config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)  # Color stream configuration

# # Start recording
# profile = pipe.start(config)

# #define the output file path and properties
# output_path = 'output.avi'
# fps = 30
# fourcc = cv.VideoWriter_fourcc(*'XVID')
# output_size = (640,480)

# #Create a video writer object to save the video
# out = cv.VideoWriter(output_path, fourcc, fps, output_size)

# start_time = t.time() # Get the current time

# while (t.time() - start_time) < 2:
#     frames = pipe.wait_for_frames()
#     color_frame = frames.get_color_frame()

#     # Convert color frame to OpenCV format
#     color_image = np.asanyarray(color_frame.get_data())
    
#     # Stops code if there's no frames captured
#     if color_frame is None: 
#         break

#     # Write the frame to the output video file
#     cv.imshow('Video', color_frame)

#     #write the frame to the output video file
#     out.write(color_image)


# # set up video capture to obtain image from saved video and detect the distance 
# cap = cv.VideoCapture(0)

# #Check if the video is open
# if not cap.isOpened():
#     print("Failed to open video capture")
#     exit()

# #capture a frame
# ret, frame = cap.read()

# if not ret:
#     print("Failed to capture a frame")
#     exit()

# #save the frame as an image file
# output_path = "reference_image.jpg"
# cv.imwrite(output_path, frame)

# # Release the video capture
# cap.release()

# print("Reference image saved successfully")

# reference_image_path = "reference_image.jpg"
# reference_image = cv.imread(reference_image_path)


# while True:
#     frames = pipe.wait_for_frames()
#     depth_frame = frames.get.depth_frame()
#     color_frame = frames.get.color_frame()

#     if not color_frame or not depth_frame:
#         continue

#     # convert the color frame to a numpy array
#     color_image = np.asanarray(color_frame.get_data())
#     depth_image = np.asanarray(depth_frame.get_data())

#     # obtains the depth value at the center of the detected object
#     center_x = int((x + w) / 2)
#     center_y = int((y + w) / 2)
#     depth_value = depth_frame.get_distance(center_x, center_y)

#     # performs distance estimation using the deoth value
#     # use camera's parameters and depth information to estimate the distance
#     focal_length_x = 617
#     focal_length_y = 617
#     principal_point_x = 319.5
#     principal_point_y = 239.5

#     # calculate the distance using the pinhole camera model
#     distance = depth_value / np.sqrt((center_x - principal_point_x)**2 / focal_length_x**2 +
#                                      (center_y - principal_point_y)**2 / focal_length_y**2 + 1)

#     # Display the distance on the image
#     cv.putText(color_image, f"Distance: {distance:.2f} meters", (10.30),
#                cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
#     cv.imshow("Distance Estimation", color_image)

#     # Break the loop if 'q' key is pressed
#     if cv.waitKey(1) & 0xFF == ord('q'):
#         break



# # Stop the pipeline and release resources
# cap.release()
# cv.destroyAllWindows()


import cv2 as cv
import numpy as np
import pyrealsense2 as rs
import time as t

# Create a pipeline to start streaming from the camera
pipe = rs.pipeline()

# Configuration prepares the camera for recording
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)  # Color stream configuration
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

# Start recording
profile = pipe.start(config)

# # Define the output file path and properties
# output_path = 'output.avi'
# fps = 30
# fourcc = cv.VideoWriter_fourcc(*'XVID')
# output_size = (640, 480)

# # Create a video writer object to save the video
# out = cv.VideoWriter(output_path, fourcc, fps, output_size)

# Start time
start_time = t.time()

while (t.time() - start_time) < 2:
    frames = pipe.wait_for_frames()
    color_frame = frames.get_color_frame()
    depth_frame = frames.get_depth_frame()

    # Convert color frame to OpenCV format
    color_image = np.asarray(color_frame.get_data())
    depth_image = np.asarray(depth_frame.get_data())

    depth_sensor = pipe.get_active_profile().get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    
    depth_image = depth_image * depth_scale

    # Display the video
    cv.imshow('Color Image', color_image)
    cv.imshow('Depth Image', depth_image)


    # Break the loop if 'q' key is pressed
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video writer and pipeline
pipe.stop()

# Set up video capture to obtain image from saved video and detect the distance
cap = cv.VideoCapture(output_path)

# Check if the video is open
if not cap.isOpened():
    print("Failed to open video capture")
    exit()

# Capture a frame
ret, frame = cap.read()

if not ret:
    print("Failed to capture a frame")
    exit()

# Save the frame as an image file
output_path = "reference_image.jpg"
cv.imwrite(output_path, frame)

# Release the video capture
cap.release()

print("Reference image saved successfully")

# Load the reference image
reference_image_path = "reference_image.jpg"
reference_image = cv.imread(reference_image_path)

# Create a pipeline to start streaming from the camera
pipe = rs.pipeline()

# Configuration prepares the camera for recording
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)  # Color stream configuration

# Start recording
profile = pipe.start(config)

while True:
    frames = pipe.wait_for_frames()
    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()

    if not color_frame or not depth_frame:
        continue

    # Convert the color frame to a numpy array
    color_image = np.asarray(color_frame.get_data())

    # Convert the depth frame to a numpy array
    depth_image = np.asarray(depth_frame.get_data())

    # Object detection and distance estimation logic here
    # ...
    # Obtain the object detection coordinates (x, y, w, h)

    # Obtain the depth value at the center of the detected object
    center_x = int((x + w) / 2)
    center_y = int((y + h) / 2)
    depth_value = depth_frame.get_distance(center_x, center_y)

    # Perform distance estimation using the depth value and camera parameters
    focal_length_x = 617
    focal_length_y = 617
    principal_point_x = 319.5
    principal_point_y = 239.5

    distance = depth_value / np.sqrt((center_x - principal_point_x)**2 / focal_length_x**2 +
                                     (center_y - principal_point_y)**2 / focal_length_y**2 + 1)

    # Display the distance on the image
    cv.putText(color_image, f"Distance: {distance:.2f} meters", (10, 30),
               cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv.imshow("Distance Estimation", color_image)

    # Break the loop if 'q' key is pressed
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# Release the pipeline and resources
pipe.stop()
cv.destroyAllWindows()
