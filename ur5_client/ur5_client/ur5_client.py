#! /usr/bin/env python3

import rclpy  # import Rospy
from rclpy.node import Node  # import Rospy Node
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor, SingleThreadedExecutor
from std_msgs.msg import String

from ur5_driver.ur5_driver import UR5 
from time import sleep

from wei_services.srv import WeiDescription 
from wei_services.srv import WeiActions  

class UR5Client(Node):
    '''
    The jointControlNode inputs data from the 'action' topic, providing a set of commands for the driver to execute. It then receives feedback, 
    based on the executed command and publishes the state of the peeler and a description of the peeler to the respective topics.
    '''
    def __init__(self, TEMP_NODE_NAME = "UR5_Client_Node"):
        '''
        The init function is neccesary for the peelerNode class to initialize all variables, parameters, and other functions.
        Inside the function the parameters exist, and calls to other functions and services are made so they can be executed in main.
        '''

        super().__init__(TEMP_NODE_NAME)
        self.node_name = self.get_name()

        self.state = "UNKNOWN"
        self.robot_status = ""
        self.action_flag = "READY"
        self.ur5 = None
        self.connect_robot()
        action_cb_group = ReentrantCallbackGroup()
        robot_state_refresher_cb_group = ReentrantCallbackGroup()
        state_cb_group = ReentrantCallbackGroup()
        description_cb_group = ReentrantCallbackGroup()

        timer_period = 0.5  # seconds

        self.statePub = self.create_publisher(String, self.node_name + '/state', 10)
        self.stateTimer = self.create_timer(timer_period, self.stateCallback, callback_group = state_cb_group)
   
        self.action_handler = self.create_service(WeiActions, self.node_name + "/action_handler", self.actionCallback, callback_group=action_cb_group)

        self.description={}
        self.descriptionSrv = self.create_service(WeiDescription, self.node_name + "/description_handler", self.descriptionCallback, callback_group=description_cb_group)

    def test(self):
        
        for i in range(3):
            sleep(20)
            # self.ur5.transfer(self.ur5.plate_exchange_1,self.ur5.plate_exchange_1)
        pass
        
    def connect_robot(self):
        
        try:
            self.ur5 = UR5()
        except Exception as err:
            self.get_logger().error(err)
        else:
            self.get_logger().info("UR5 connected")
            
    def stateCallback(self):
        '''
        Publishes the peeler state to the 'state' topic. 
        '''
        msg = String()

        try:
            self.state = self.ur5.get_movement_state()

        except Exception as err:
            self.get_logger().error("ROBOT IS NOT RESPONDING! ERROR: " + str(err))
            self.state = "UR5 CONNECTION ERROR"

        msg.data = 'State: %s' % self.state
        self.statePub.publish(msg)
        self.get_logger().info('Publishing: "%s"' % msg.data)
        # self.state = "READY"

    def descriptionCallback(self, request, response):
        """The descriptionCallback function is a service that can be called to showcase the available actions a robot
        can preform as well as deliver essential information required by the master node.

        Parameters:
        -----------
        request: str
            Request to the robot to deliver actions
        response: str
            The actions a robot can do, will be populated during execution

        Returns
        -------
        str
            The robot steps it can do
        """
        response.description_response = str(self.description)

        return response

    def actionCallback(self, request, response):
        '''
        The actionCallback function is a service that can be called to execute the available actions the robot
        can preform.
        '''
        
        if request.action_handle=='transfer':
            self.state = "BUSY"
            self.stateCallback()
            vars = eval(request.vars)
            print(vars)

            if 'pos1' not in vars.keys() or 'pos2' not in vars.keys():
                print('vars wrong')
                return 

            pos1 = vars.get('pos1')
            print(pos1)
            pos2 = vars.get('pos2')
            print(pos2)

            self.ur5.transfer(pos1, pos2)
            

        self.state = "COMPLETED"

        return response


def main(args = None):

    rclpy.init(args=args)  # initialize Ros2 communication

    try:
        ur5_client = UR5Client()
        executor = MultiThreadedExecutor()
        executor.add_node(ur5_client)

        try:
            ur5_client.get_logger().info('Beginning client, shut down with CTRL-C')
            executor.spin()
        except KeyboardInterrupt:
            ur5_client.get_logger().info('Keyboard interrupt, shutting down.\n')
        finally:
            executor.shutdown()
            ur5_client.ur5.disconnect_ur()
            ur5_client.destroy_node()
    finally:
        rclpy.shutdown()

if __name__ == '__main__':
    main()