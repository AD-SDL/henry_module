#!/usr/bin/env python3

import threading

from multiprocessing.connection import wait
from time import sleep
from copy import deepcopy

from ur5_driver.ur_dashboard import UR_DASHBOARD
import ur5_driver.robotiq_gripper as robotiq_gripper
from urx import Robot, RobotException

class UR5(UR_DASHBOARD):
    

    def __init__(self, IP:str = "146.137.240.38", PORT: int = 29999):

        super().__init__(IP=IP, PORT=PORT)

        self.initialize() # Initilialize the robot

        # ur5 SETUP:
        self.ur5 = None
        self.connect_ur()

        self.acceleration = 2
        self.velocity = 2
        self.robot_current_joint_angles = None
        # self.test1
        self.module_entry = [-0.1828145484680406, 0.1501917529215074, 0.4157045667286946, -0.014753354925067616, -3.133785224432585, -0.01020982277167234]
        self.module_entry_joint = [-1.3963525930987757, -2.1945158443846644, 2.1684568564044397, -1.5495260164937754, -1.5337546507464808, 3.2634336948394775]
        self.home = [-0.13358071546889347, -0.009673715752021885, 0.5890782758304143, -0.014566051910791617, -3.133734935087693, -0.010359747956377084]
        self.home_joint = [-1.355567757283346, -2.5413090191283167, 1.8447726408587855, -0.891581193809845, -1.5595606009112757, 3.3403327465057373]
        self.plate_exchange_1_above = [-0.18284724105645211, 0.7914820291585895, 0.41175512257988434, -0.014545475433050672, -3.1337759450718, -0.010278634391729295]
        self.plate_exchange_1 = [-0.1828537989205587, 0.7914917511283945, 0.390542100409092, -0.014571172649734884, -3.133719848650817, -0.010138239501312422]

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
        

        self.gripper_close = 130 # 0-255 (255 is closed)
        self.griper_open = 0
        self.gripper_speed = 150 # 0-255
        self.gripper_force = 0 # 0-255

        print('Opening gripper...')
        self.gripper.move_and_wait_for_pos(self.griper_open, self.gripper_speed, self.gripper_force)

    def connect_ur(self):
        """
        Description: Create conenction to the UR robot
        """

        for i in range(10):
            try:
                self.ur5 = Robot(self.IP)
                sleep(2)

            except Exception:
                print("Trying robot connection ...")
            else:
                print('Successful ur5 connection')
                break

    def disconnect_ur(self):
        """
        Description: Disconnects the socket connection with the UR robot
        """
        self.ur5.close()
        print("Robot connection is closed.")

    def get_joint_angles(self):
        
        return self.ur5.getj()
    
    def get_cartesian_coordinates(self):
        
        return self.ur5.getl()
    
    def get_movement_state(self):
        current_location = self.get_joint_angles()
        
        if self.robot_current_joint_angles == current_location:
            movement_state = "READY"
        else:
            movement_state = "BUSY"

        self.robot_current_joint_angles = current_location

        return movement_state

    def pick(self, pick_goal):

        '''Pick up from first goal position'''

        above_goal = deepcopy(pick_goal)
        above_goal[2] += 0.05

        print('Moving to home position')
        # self.ur5.movel(self.home, self.acceleration, self.velocity)
        self.ur5.movej(self.home_joint, self.acceleration, self.velocity)
        # sleep(1)
        print("Moving to the module entry location")
        # self.ur5.movel(self.module_entry, self.acceleration, self.velocity)
        self.ur5.movej(self.module_entry_joint, self.acceleration, self.velocity)
        # sleep(1)
        print('Moving to above goal position')
        self.ur5.movel(above_goal, self.acceleration, self.velocity)
        # sleep(1)
        print('Moving to goal position')
        self.ur5.movel(pick_goal, self.acceleration, self.velocity)
        # sleep(1)
        print('Closing gripper')
        self.gripper.move_and_wait_for_pos(self.gripper_close, self.gripper_speed, self.gripper_force)
        # sleep(1)
        print('Moving back to above goal position')
        self.ur5.movel(above_goal, self.acceleration, self.velocity)
        # sleep(1)
        print("Moving to the module entry location")
        # self.ur5.movel(self.module_entry, self.acceleration, self.velocity)
        self.ur5.movej(self.module_entry_joint, self.acceleration, self.velocity)
        # sleep(1)
        print('Moving to home position')
        # self.ur5.movel(self.home, self.acceleration, self.velocity)
        self.ur5.movej(self.home_joint, self.acceleration, self.velocity)
        # sleep(1)

    def place(self, place_goal):

        '''Place down at second goal position'''

        above_goal = deepcopy(place_goal)
        above_goal[2] += 0.05

        print('Moving to home position')
        # self.ur5.movel(self.home, self.acceleration, self.velocity)
        self.ur5.movej(self.home_joint, self.acceleration, self.velocity)
        # sleep(1)
        print("Moving to the module entry location")
        # self.ur5.movel(self.module_entry, self.acceleration, self.velocity)
        self.ur5.movej(self.module_entry_joint, self.acceleration, self.velocity)
        # sleep(1)
        print('Moving to above goal position')
        self.ur5.movel(above_goal, self.acceleration, self.velocity)
        # sleep(1)
        print('Moving to goal position')
        self.ur5.movel(place_goal, self.acceleration, self.velocity)
        # sleep(1)
        print('Opennig gripper')
        self.gripper.move_and_wait_for_pos(self.griper_open, self.gripper_speed, self.gripper_force)
        # sleep(1)
        print('Moving back to above goal position')
        self.ur5.movel(above_goal, self.acceleration, self.velocity)
        # sleep(1)
        print("Moving to the module entry location")
        # self.ur5.movel(self.module_entry, self.acceleration, self.velocity)
        self.ur5.movej(self.module_entry_joint, self.acceleration, self.velocity)
        # sleep(1)
        print('Moving to home position')
        # self.ur5.movel(self.home, self.acceleration, self.velocity)
        self.ur5.movej(self.home_joint, self.acceleration, self.velocity)
        # sleep(1)
        
    def transfer(self, pos1, pos2):
        ''''''
        self.ur5.set_tcp((0, 0, 0, 0, 0, 0))
        self.pick(pos1)
        self.place(pos2)
        # self.disconnect_ur()s
        print('Finished transfer')

if __name__ == "__main__":

    # pos1= [-0.22575, -0.65792, 0.39271, 2.216, 2.196, -0.043]
    # pos2= [0.22575, -0.65792, 0.39271, 2.216, 2.196, -0.043]
    robot = UR5()
    # robot.ur5.movej(robot.home_joint, robot.acceleration, robot.velocity)
    robot.transfer(robot.plate_exchange_1,robot.plate_exchange_1)
    # robot.ur5.set_tcp((0, 0, 0.1, 0, 0, 0))
    # robot.ur5.set_payload(2, (0, 0, 0.1))
    sleep(0.2)  #leave some time to robot to process the setup commands    # robot.transfer(pos2,pos1)
    # print(robot.ur5.getl())
    # for i in range(10):
    #     print(robot.get_movement_state())
    #     sleep(0.5)
    # print(robot.get_joint_angles())
    # print(robot.get_cartesian_coordinates())
    robot.disconnect_ur()
    print('end')
