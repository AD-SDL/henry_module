#!/usr/bin/env python3

import threading

from multiprocessing.connection import wait

import time

import robotiq_gripper
from UR_12idb.robUR3 import UR3
# from urx import Robot
from copy import deepcopy
from ur_dashboard import UR_DASHBOARD
# from UR_12idb.urcamera import camera

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
        self.home_back = [-5.423735324536459, -1.7657300434508265, 2.6696303526507776, -0.908503608112671, 0.8667137622833252, 0.0054948958568274975]
        self.home_left = [-4.814536158238546, -1.4853655856898804, 2.1723483244525355, -0.7514525812915345, 2.992141008377075, -0.0323408285724085]
        self.home_right = [-5.429234568272726, -1.394191862349846, 2.1012232939349573, -0.7222079199603577, -0.7285550276385706, 0.0160035602748394]

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

        direction = pick_goal[0]
        goal = pick_goal[1]

        # Change arm to face the correct directly (front, back, left, or right)
        if direction == 'back':
            home = self.home_back
        elif direction == 'left':
            home = self.home_left
        elif direction == 'right':
            home = self.home_right
        else:
            print('Orientation does not exist')
            return

        # Calculate position above goal
        above_goal = deepcopy(goal)
        above_goal[2] += 0.05

        print('Moving to home position')
        self.ur5.robot.movej(home, self.acceleration, self.velocity)

        print('Moving to above goal position')
        self.ur5.robot.movel(above_goal, self.acceleration, self.velocity)

        print('Moving to goal position')
        self.ur5.robot.movel(goal, self.acceleration, self.velocity)

        print('Closing gripper')
        self.gripper.move_and_wait_for_pos(self.gripper_close, self.gripper_speed, self.gripper_force)

        print('Moving back to above goal position')
        self.ur5.robot.movel(above_goal, self.acceleration, self.velocity)

        print('Moving to home position')
        self.ur5.robot.movej(home, self.acceleration, self.velocity)

    

    def place(self, place_goal):

        '''Place down at second goal position'''

        direction = place_goal[0]
        goal = place_goal[1]

        # Change arm to face the correct directly (front, back, left, or right)
        if direction == 'back':
            home = self.home_back
        elif direction == 'left':
            home = self.home_left
        elif direction == 'right':
            home = self.home_right
        else:
            print('Orientation does not exist')
            return

        # Calculate position above goal
        above_goal = deepcopy(goal)
        above_goal[2] += 0.05

        print('Moving to home position')
        self.ur5.robot.movej(home, self.acceleration, self.velocity)

        print('Moving to above goal position')
        self.ur5.robot.movel(above_goal, self.acceleration, self.velocity)

        print('Moving to goal position')
        self.ur5.robot.movel(goal, self.acceleration, self.velocity)

        print('Opennig gripper')
        self.gripper.move_and_wait_for_pos(self.griper_open, self.gripper_speed, self.gripper_force)

        print('Moving back to above goal position')
        self.ur5.robot.movel(above_goal, self.acceleration, self.velocity)

        print('Moving to home position')
        self.ur5.robot.movej(home, self.acceleration, self.velocity)

        if direction != 'back':
            print('Moving to BACK home position')
            self.ur5.robot.movej(self.home_back, self.acceleration, self.velocity)


    def transfer(self, pos1, pos2):
        ''''''
        self.pick(pos1)
        self.place(pos2)
        print('Finished transfer')


    def close(self):
        self.clear_operational_mode()
        self.ur5.robot.close()


if __name__ == "__main__":

    pos_front1 = ['back', [ 0.19699137, -0.65753703, 0.17121313, 1.58650296, 0.00313809, 0.00971325]]
    pos_front2 = ['back', [-0.20288636, -0.65752902, 0.17121757, 1.58646952, 0.00336391, 0.0097106 ]]

    pos_right1 = ['right', [0.60434288, -0.27237252, 0.20709442, 1.04564069, 1.28511247, 1.23740526]]

    pos_left1 = ['left', [-0.47370519, -0.26544203, 0.21667274, 1.19952641, -1.17216141, -1.16812065]]


    robot = UR5()


    # To get joint and cartesian position of arm:
    print(robot.ur5.getj())
    print(robot.ur5.get_xyz())

    robot.transfer(pos_front1,pos_left1)

    # To capture camera frame and save to file:
    # print(robot.ur5.camera.capture())
    # robot.ur5.camera.save()
    
    # robot.ur5.robot.movel(robot.home, robot.acceleration, robot.velocity)
    # To center camera onto a QR code:
    # robot.ur5.bring_QR_to_camera_center(referenceName='QR')

    robot.close()
    print('end')
