import robotiq_gripper


print('Creating gripper...')
gripper = robotiq_gripper.RobotiqGripper()
print('Connecting to gripper...')

gripper.connect('192.168.50.82', 63352)

if gripper.is_active():
    print('Gripper already active')
else:
    print('Activating gripper...')
    gripper.activate()


gripper_close = 110 # 0-255 (255 is closed)
gripper_open = 0
gripper_speed = 150 # 0-255
gripper_force = 0 # 0-255

print('Closing gripper...')
gripper.move_and_wait_for_pos(gripper_close, gripper_speed, gripper_force)

print('Opening gripper...')
gripper.move_and_wait_for_pos(gripper_open, gripper_speed, gripper_force)