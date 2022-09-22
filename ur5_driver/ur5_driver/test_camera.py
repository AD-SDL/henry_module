from UR_12idb import urcamera
import time
import cv2

# This tests shows the camera if directly plugged into the computer
# You cannot change focus if getting image from the camera server

camera = urcamera.camera(device=0)


camera.capture()

# data, rectcoord, qrsize, dist
while True:
    camera.autofocus()
    camera.capture()
    cv2.imshow('camera', camera.image)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
    camera.decode()
    camera.analyzeroll_QR()

# camera.save("test_camera")