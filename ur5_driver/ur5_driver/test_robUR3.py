from UR_12idb.robUR3 import UR3
import time
from copy import deepcopy

i = 1
while True:
    try:
        rob = UR3(name = '192.168.50.82', device = 0)
        time.sleep(0.2)
        print('Successful ur5 connection on attempt #{}'.format(i))
        break
    except:
        print('Failed attempt #{}'.format(i))
        i+=1

# rob.camera.focus(450)

# # rob.bring_QR_to_camera_center(referenceName='QR')
# rob.camera.capture()
# rob.camera.decode()
# rob.relocate_camera()
# rob.tilt_align()
rob.camera.focus(450)
# current_pos = rob.robot.getl()
# rob.robot.movel((0.0, -0.200, 0.59262, 2.247, 2.196, 0.0), 1.0, 0.2)
while True:
    rob.align_ry(1.0, 0.2)
    
# print(len(rob.camera.QRcoordinates))
rob.terminate()

