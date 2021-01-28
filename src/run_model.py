#!/usr/bin/env python

from numpy.lib.financial import rate
import rospy
import rospkg

import numpy

from utils.dqn import DEVICE, QRunningAgent

from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from std_srvs.srv import Empty

RUNNING_TIME = 0.08
UPDATE_RATE = 50
MIN_RANGE = 0.2
MAX_LASER_VALUE = 3.5
MIN_LASER_VALUE = 0

def do_step(action):

    rate = rospy.Rate(UPDATE_RATE)

    if action == 0: #FORWARD
        linear_speed = 0.3
        angular_speed = 0.0
        print('Forward')
    elif action == 1: #LEFT
        linear_speed = 0.05
        angular_speed = 0.3
        print('Left')
    elif action == 2: #RIGHT
        linear_speed = 0.05
        angular_speed = -0.3
        print('Right')
    else: #NOP
        linear_speed = 0.0
        angular_speed = 0.0

    cmd_vel_value = Twist()
    cmd_vel_value.linear.x = linear_speed
    cmd_vel_value.angular.z = angular_speed
    rospy.Publisher('/cmd_vel', Twist, queue_size=1).publish(cmd_vel_value)
    
    start_wait_time = rospy.get_rostime().to_sec()
    while (not rospy.is_shutdown()) and (not rospy.get_rostime().to_sec() - start_wait_time > 0.08):
        rate.sleep()

    cmd_vel_value = Twist()
    cmd_vel_value.linear.x = 0.0
    cmd_vel_value.angular.z = 0.0

    ranges = get_ranges()

    return ranges

def is_done(ranges):
    done = False
    if MIN_RANGE > min(ranges) > 0:
        done = True
    return done

def get_ranges():

    laser_scan = None
    
    while laser_scan is None and not rospy.is_shutdown():
        try:
            laser_scan = rospy.wait_for_message('/scan', LaserScan, timeout=1.0)
        except:
            pass
    
    ranges = []
    for item in laser_scan.ranges:
        if numpy.isinf(item):
            ranges.append(MAX_LASER_VALUE)
        elif numpy.isnan(item):
            ranges.append(MIN_LASER_VALUE)
        else:
            ranges.append(item)

    return ranges

if __name__ == "__main__":
    rospy.init_node('turtlebot3_run', anonymous=True, log_level=rospy.INFO)

    rospack = rospkg.RosPack()
    pkg_path = rospack.get_path('turtlebot3_training')
    model_path = pkg_path + '/model/20210128_1000/model.ph'
    
    qrun = QRunningAgent(model_path)

    crash = False

    states = get_ranges()

    while not crash:
        action = qrun.choose_action(states)
        next_states = do_step(action)
        crash = is_done(next_states)
        if not crash:
            states = next_states
        else:
            rospy.ServiceProxy('/gazebo/reset_simulation', Empty)
            rospy.loginfo('Bot crashed!')

