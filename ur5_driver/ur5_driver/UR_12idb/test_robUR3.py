import robUR3
import time

i = 1
while True:
    try:
        rob = robUR3.UR3(name = '192.168.50.82', device = 2)
        time.sleep(0.2)
        print('Successful ur5 connection on attempt #{}'.format(i))
        break
    except:
        print('Failed attempt #{}'.format(i))
        i+=1
        time.sleep(2)

rob.camera.focus(450)

# rob.bring_QR_to_camera_center(referenceName='QR')
rob.camera.capture()
rob.camera.decode()
rob.relocate_camera()
# rob.tilt_align()

rob.terminate()

