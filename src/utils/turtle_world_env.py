import rospy
import numpy
from gym import spaces
from .turtle_env import TurtleBot3Env
from gym.envs.registration import register
from geometry_msgs.msg import Vector3

# The path is __init__.py of openai_ros, where we import the TurtleBot2MazeEnv directly
timestep_limit_per_episode = 10000 # Can be any Value

# register the training enviroment in the gym as an available one
reg = register(
    id = 'TurtleBot3World-v0',
    entry_point = "utils.turtle_world_env:TurtleBot3WorldEnv",
    max_episode_steps = 1000, 
)

class TurtleBot3WorldEnv(TurtleBot3Env):
    def __init__(self):
        """
        This Task Env is designed for having the TurtleBot3 in the turtlebot3 world
        closed room with columns.
        It will learn how to move around without crashing.
        """
        
        # Only variable needed to be set here
        number_actions = rospy.get_param('/n_actions')
        self.action_space = spaces.Discrete(number_actions)
        
        # We set the reward range, which is not compulsory but here we do it.
        self.reward_range = (-numpy.inf, numpy.inf)
        
        
        #number_observations = rospy.get_param('n_observations')
        """
        We set the Observation space for the 6 observations
        cube_observations = [
            round(current_disk_roll_vel, 0),
            round(y_distance, 1),
            round(roll, 1),
            round(pitch, 1),
            round(y_linear_speed,1),
            round(yaw, 1),
        ]
        """
        
        # Actions and Observations
        self.linear_forward_speed = rospy.get_param('/linear_forward_speed')
        self.linear_turn_speed = rospy.get_param('/linear_turn_speed')
        self.angular_speed = rospy.get_param('/angular_speed')
        self.init_linear_forward_speed = rospy.get_param('/init_linear_forward_speed')
        self.init_linear_turn_speed = rospy.get_param('/init_linear_turn_speed')
        
        self.new_ranges = rospy.get_param('/new_ranges')
        self.min_range = rospy.get_param('/min_range')
        self.max_laser_value = rospy.get_param('/max_laser_value')
        self.min_laser_value = rospy.get_param('/min_laser_value')
        self.max_linear_aceleration = rospy.get_param('/max_linear_aceleration')
        self.running_time = rospy.get_param('/running_time')
        
        
        # We create two arrays based on the binary values that will be assigned
        # In the discretization method.
        laser_scan = self._check_laser_scan_ready()
        num_laser_readings = len(laser_scan.ranges)/self.new_ranges
        high = numpy.full(int(num_laser_readings), self.max_laser_value)
        low = numpy.full(int(num_laser_readings), self.min_laser_value)
        
        # We only use two integers
        self.observation_space = spaces.Box(low, high)
        
        rospy.logdebug("ACTION SPACES TYPE===>"+str(self.action_space))
        rospy.logdebug("OBSERVATION SPACES TYPE===>"+str(self.observation_space))
        
        # Rewards
        self.forwards_reward = rospy.get_param("/forwards_reward")
        self.turn_reward = rospy.get_param("/turn_reward")
        self.end_episode_points = rospy.get_param("/end_episode_points")

        self.cumulated_steps = 0.0

        # Here we will add any init functions prior to starting the MyRobotEnv
        super(TurtleBot3WorldEnv, self).__init__()

    def _set_init_pose(self):
        """Sets the Robot in its init pose
        """
        self.move_base( self.init_linear_forward_speed,
                        self.init_linear_turn_speed,
                        running_time=self.running_time,
                        epsilon=0.05,
                        update_rate=10)

        return True


    def _init_env_variables(self):
        """
        Inits variables needed to be initialised each time we reset at the start
        of an episode.
        :return:
        """
        # For Info Purposes
        self.cumulated_reward = 0.0
        # Set to false Done, because its calculated asyncronously
        self._episode_done = False


    def _set_action(self, action):
        """
        This set action will Set the linear and angular speed of the turtlebot2
        based on the action number given.
        :param action: The action integer that set s what movement to do next.
        """
        
        rospy.logdebug("Start Set Action ==>"+str(action))
        # We convert the actions to speed movements to send to the parent class CubeSingleDiskEnv
        if action == 0: #FORWARD
            linear_speed = self.linear_forward_speed
            angular_speed = 0.0
            self.last_action = "FORWARDS"
        elif action == 1: #LEFT
            linear_speed = self.linear_turn_speed
            angular_speed = self.angular_speed
            self.last_action = "TURN_LEFT"
        elif action == 2: #RIGHT
            linear_speed = self.linear_turn_speed
            angular_speed = -1*self.angular_speed
            self.last_action = "TURN_RIGHT"
        
        # We tell TurtleBot2 the linear and angular speed to set to execute
        self.move_base(linear_speed, angular_speed, running_time=self.running_time, epsilon=0.05, update_rate=40)
        
        rospy.logdebug("END Set Action ==>"+str(action))

    def _get_obs(self):
        """
        Here we define what sensor data defines our robots observations
        To know which Variables we have acces to, we need to read the
        TurtleBot2Env API DOCS
        :return:
        """
        rospy.logdebug("Start Get Observation ==>")
        # We get the laser scan data
        laser_scan = self.get_laser_scan()
        
        discretized_observations = self.discretize_scan_observation(    laser_scan,
                                                                        self.new_ranges
                                                                        )

        rospy.logdebug("Observations==>"+str(discretized_observations))
        rospy.logdebug("END Get Observation ==>")
        return discretized_observations
        

    def _is_done(self, observations):
        
        if self.min_range > min(observations) > 0:
            rospy.logdebug("TurtleBot3 is Too Close to wall==>")
            self._episode_done = True
        else:
            rospy.logdebug("TurtleBot3 is NOT close to a wall ==>")
            
        # Now we check if it has crashed based on the imu
        # imu_data = self.get_imu()
        # rospy.logdebug("linear_acceleration ==>"+str(imu_data.linear_acceleration))
        # linear_acceleration_magnitude = self.get_vector_magnitude(imu_data.linear_acceleration)
        # rospy.logdebug("linear_acceleration_magnitude ==>"+str(linear_acceleration_magnitude))
        # if linear_acceleration_magnitude > self.max_linear_aceleration:
        #     rospy.logerr("TurtleBot3 Crashed==>"+str(linear_acceleration_magnitude)+">"+str(self.max_linear_aceleration))
        #     self._episode_done = True
        # else:
        #     rospy.logdebug("DIDNT crash TurtleBot3 ==>"+str(linear_acceleration_magnitude)+">"+str(self.max_linear_aceleration))        

        return self._episode_done

    def _compute_reward(self, observations, done):

        if not done:
            if self.last_action == "FORWARDS":
                reward = self.forwards_reward
            else:
                reward = self.turn_reward
        else:
            reward = -1*self.end_episode_points


        rospy.logdebug("reward=" + str(reward))
        self.cumulated_reward += reward
        rospy.logdebug("Cumulated_reward=" + str(self.cumulated_reward))
        self.cumulated_steps += 1
        rospy.logdebug("Cumulated_steps=" + str(self.cumulated_steps))
        
        return reward


    # Internal TaskEnv Methods
    
    def discretize_scan_observation(self,data,new_ranges):
        """
        Discards all the laser readings that are not multiple in index of new_ranges
        value.
        """
        self._episode_done = False
        
        discretized_ranges = []
        mod = len(data.ranges)/new_ranges
        #mod = new_ranges
                
        #rospy.logdebug("data=" + str(data.ranges))
        rospy.logdebug("new_ranges=" + str(new_ranges))
        rospy.logdebug("mod=" + str(mod))

        for i in range(int(mod)):
            n_low = i*new_ranges
            n_high = (i+1)*new_ranges
            batch = []        
            for item in data.ranges[n_low: n_high]:
                if numpy.isinf(item):
                    batch.append(self.max_laser_value)
                elif numpy.isnan(item):
                    batch.append(self.min_laser_value)
                else:
                    batch.append(item)
            
            discretized_ranges.append(round(min(batch), 1))

        # for i, item in enumerate(data.ranges):
        #     if (i%mod==0):
        #         if item == float ('Inf') or numpy.isinf(item):
        #             discretized_ranges.append(self.max_laser_value)
        #         elif numpy.isnan(item):
        #             discretized_ranges.append(self.min_laser_value)
        #         else:
        #             discretized_ranges.append(round(item, 1))
                    
        #         if (self.min_range > item > 0):
        #             rospy.logerr("done Validation >>> item=" + str(item)+"< "+str(self.min_range))
        #             self._episode_done = True
        #         else:
        #             rospy.logdebug("NOT done Validation >>> item=" + str(item)+ " ("+str(i) +") < "+str(self.min_range))

        return discretized_ranges
        
        
    def get_vector_magnitude(self, vector):
        """
        It calculated the magnitude of the Vector3 given.
        This is usefull for reading imu accelerations and knowing if there has been 
        a crash
        :return:
        """
        contact_force_np = numpy.array((vector.x, vector.y, vector.z))
        force_magnitude = numpy.linalg.norm(contact_force_np)

        return force_magnitude