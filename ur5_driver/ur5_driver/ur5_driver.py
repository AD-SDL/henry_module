#!/usr/bin/env python3

import threading

from multiprocessing.connection import wait

import time

# import ur5_driver.robotiq_gripper as robotiq_gripper
import robotiq_gripper
from UR_12idb.robUR3 import UR3
from urx import Robot
from copy import deepcopy
from ur_dashboard import UR_DASHBOARD
from UR_12idb.urcamera import camera

class UR5(UR_DASHBOARD):
    
    '''camera_index: need to specify camera when using camera directly plugged into computer.
    0 is default, 2 is the next camera'''

    def __init__(self, IP:str = "192.168.50.82", PORT: int = 29999, camera_index = 0):

        super().__init__(IP=IP, PORT=PORT)


        self.initialize() # Initilialize the robot


        # UR5 SETUP:
        i = 1
        while True:
            try:
                self.ur5 = UR3(name = self.IP, device = camera_index)
                time.sleep(0.2)
                print('Successful ur5 connection on attempt #{}'.format(i))
                break
            except:
                print('Failed attempt #{}'.format(i))
                i+=1

        # Variables to control the arm's speed
        self.acceleration = 1.0
        self.velocity = 0.2

        # Joint position of arm's home position
        self.home = [-5.432596866284506, -1.7744094334044398, 2.659326140080587, -0.8687126797488709, 0.8495127558708191, -0.0038750807391565445]


        # GRIPPER SETUP:
        print('Creating gripper...')
        self.gripper = robotiq_gripper.RobotiqGripper()
        print('Connecting to gripper...')
        
        self.gripper.connect(self.IP, 63352)

        if self.gripper.is_active():
            print('Gripper already active')
        else:
            print('Activating gripper...')
            self.gripper.activate()

        # Variables to control gripper
        self.gripper_close = 110 # 0-255 (255 is closed)
        self.griper_open = 0
        self.gripper_speed = 150 # 0-255
        self.gripper_force = 0 # 0-255

        print('Opening gripper...')
        self.gripper.move_and_wait_for_pos(self.griper_open, self.gripper_speed, self.gripper_force)


    def pick(self, pick_goal):

        '''Pick up from first goal position'''

        above_goal = deepcopy(pick_goal)
        above_goal[2] += 0.05

        print('Moving to home position')
        self.ur5.robot.movej(self.home, self.acceleration, self.velocity)

        print('Moving to above goal position')
        self.ur5.robot.movel(above_goal, self.acceleration, self.velocity)

        print('Moving to goal position')
        self.ur5.robot.movel(pick_goal, self.acceleration, self.velocity)

        print('Closing gripper')
        self.gripper.move_and_wait_for_pos(self.gripper_close, self.gripper_speed, self.gripper_force)

        print('Moving back to above goal position')
        self.ur5.robot.movel(above_goal, self.acceleration, self.velocity)

        print('Moving to home position')
        self.ur5.robot.movej(self.home, self.acceleration, self.velocity)

    

    def place(self, place_goal):

        '''Place down at second goal position'''

        above_goal = deepcopy(place_goal)
        above_goal[2] += 0.05

        print('Moving to home position')
        self.ur5.robot.movej(self.home, self.acceleration, self.velocity)

        print('Moving to above goal position')
        self.ur5.robot.movel(above_goal, self.acceleration, self.velocity)

        print('Moving to goal position')
        self.ur5.robot.movel(place_goal, self.acceleration, self.velocity)

        print('Opennig gripper')
        self.gripper.move_and_wait_for_pos(self.griper_open, self.gripper_speed, self.gripper_force)

        print('Moving back to above goal position')
        self.ur5.robot.movel(above_goal, self.acceleration, self.velocity)

        print('Moving to home position')
        self.ur5.robot.movej(self.home, self.acceleration, self.velocity)


    def transfer(self, pos1, pos2):
        ''''''
        self.pick(pos1)
        self.place(pos2)
        print('Finished transfer')


    def close(self):
        self.clear_operational_mode()
        self.ur5.robot.close()


if __name__ == "__main__":

    pos1= [ 0.19699137, -0.65753703, 0.17121313, 1.58650296, 0.00313809, 0.00971325]
    pos2= [-0.20288636, -0.65752902, 0.17121757, 1.58646952, 0.00336391, 0.0097106 ]
    robot = UR5()

    # print(robot.ur5.getj())
    # print(robot.ur5.get_xyz())

    robot.transfer(pos1,pos2)
    robot.transfer(pos2,pos1)

    # print(robot.ur5.camera.capture())
    # robot.ur5.camera.save()
    
    # robot.ur5.robot.movel(robot.home, robot.acceleration, robot.velocity)
    # robot.ur5.bring_QR_to_camera_center(referenceName='QR')

    robot.close()
    print('end')
