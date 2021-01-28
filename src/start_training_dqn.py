#!/usr/bin/env python
import sys
import os

from rospy.core import rospyinfo
cwd = os.getcwd()
if cwd not in sys.path:
    sys.path.append(cwd)
import time
import random
import collections
import numpy as np
# Gym
import gym
from gym import wrappers
from utils.turtle_world_env import TurtleBot3WorldEnv
# DQN
from utils.dqn import QLearningAgent
# ROS
import rospy
import rospkg
# Plot
from utils.liveplot import LivePlot

if __name__ == "__main__":
    
    rospy.init_node('turtlebot3_gym', anonymous=True, log_level=rospy.INFO)

    # Create the Gym environment
    env = gym.make('TurtleBot3World-v0')

    # Set the logging system and set path
    rospack = rospkg.RosPack()
    pkg_path = rospack.get_path('turtlebot3_training')
    model_path = pkg_path + '/model/model.ph'
    param_path = pkg_path + '/model/param.json'
    outdir = pkg_path + '/training_results'
    env = wrappers.Monitor(env, outdir, force=True)

    # Create plotter object
    plotter = LivePlot(outdir)

    # Loads parameters from the ROS param server
    # Parameters are stored in a yaml file inside the config directory
    # They are loaded at runtime by the launch file
    lr = rospy.get_param("/learning_rate")
    gamma = rospy.get_param("/gamma")
    epsilon = rospy.get_param("/epsilon")
    epsilon_discount = rospy.get_param("/epsilon_discount")
    nepisodes = rospy.get_param("/nepisodes")
    nsteps = rospy.get_param("/nsteps")
    resume = rospy.get_param("/resume")
    action_size = env.action_space.n
    state_size = env.observation_space.shape[0]

    # Create DQN Agent
    dqn = QLearningAgent(state_size, action_size, epsilon, lr, gamma)

    # check if to resume training
    if resume:
        dqn.load(model_path)
    
    initial_epsilon = dqn.epsilon
    last_time_steps = np.ndarray(0)
    highest_reward = 0
    start_time = time.time()

    # Starts the main training loop: the one about the episodes to do
    for x in range(nepisodes):
        rospy.loginfo("############### START EPISODE=>" + str(x))

        actions = []
        cumulated_reward = 0
        done = False

        # Now We return directly the stringuified observations called state
        state = env.reset()

        # for each episode, we test the robot for nsteps
        for i in range(nsteps):
            rospy.logdebug("###################### Start Step...[" + str(i) + "]")

            # Pick an action based on the current state
            action, action_mode = dqn.choose_action(state)

            # Execute the action in the environment and get feedback
            rospy.logdebug("Action to Perform >> " + str(action))
            next_state, reward, done, info = env.step(action)
            next_state = np.array(next_state)
            rospy.logdebug("END Step...")
            rospy.logdebug("Reward ==> " + str(reward))
            cumulated_reward += reward
            if highest_reward < cumulated_reward:
                highest_reward = cumulated_reward

            # Make the algorithm learn based on the results
            rospy.logdebug("# state we were=>" + str(state))
            rospy.logdebug("# action that we took=>" + str(action))
            rospy.logdebug("# reward that action gave=>" + str(reward))
            rospy.logdebug("# episode cumulated_reward=>" + str(cumulated_reward))
            rospy.logdebug("# State in which we will start next step=>" + str(next_state))
            dqn.learn(state, action, reward, next_state)

            actions.append((action, action_mode))

            if not (done):
                state = next_state
            else:
                rospy.logdebug("DONE")
                last_time_steps = np.append(last_time_steps, [int(i + 1)])
                break

            rospy.logdebug("###################### END Step...[" + str(i) + "]")

        m, s = divmod(int(time.time() - start_time), 60)
        h, m = divmod(m, 60)

        rospy.loginfo("Actions: " + str(collections.Counter(actions)))
        rospy.loginfo(("EP: " + str(x + 1) + " - [alpha: " + str(round(dqn.lr, 2)) + " - gamma: " + str(
            round(dqn.gamma, 2)) + " - epsilon: " + str(round(dqn.epsilon, 2)) + "] - Reward: " + str(
            cumulated_reward) + "     Time: %d:%02d:%02d" % (h, m, s)))


        dqn.epsilon = max(dqn.epsilon * epsilon_discount, 0.05)
        if x % 10:
            dqn.save(model_path)
        plotter.plot(env)

    rospy.loginfo(("\n|" + str(nepisodes) + "|" + str(dqn.lr) + "|" + str(dqn.gamma) + "|" + str(
        initial_epsilon) + "*" + str(epsilon_discount) + "|" + str(highest_reward) + "| PICTURE |"))

    l = last_time_steps.tolist()
    l.sort()

    env.close()
