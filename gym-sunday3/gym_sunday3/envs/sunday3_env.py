import math
from typing import Optional, Union

from pyfirmata import Arduino, util

import numpy as np
import random

import gym
from gym import logger, spaces
from gym.error import DependencyNotInstalled

import time

#Arrduino online


Sensor1 = 10
Sensor2 = 12

time.sleep(2)
#servo = board.get_pin('d:9:s')  



class sunday3(gym.Env[np.ndarray, Union[int, np.ndarray]]):


    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(self):
            

        
        self.battery = 10
        
        
        self.solar_panel_max = 100
        self.solar_panel_min = 0
        
        self.humidity_sensor_max = 100
        self.humidity_sensor_min = 0

        # Angle limit set to 2 * theta_threshold_radians so failing observation
        # is still within bounds.
        high = np.array(
            [
                self.solar_panel_max,
                self.humidity_sensor_max,
 
            ],
            dtype=np.float32,
        )
        
        low = np.array(
            [
                self.solar_panel_min,
                self.humidity_sensor_min,
 
            ],
            dtype=np.float32,
        )


        self.action_space = spaces.Discrete(3) #rev: 3 actions
        self.observation_space = spaces.Box(low, high, dtype=np.float32)

        # self.screen = None
        # self.clock = None
        # self.isopen = True
        self.state = None

        self.steps_beyond_done = None

    def step(self, action):
        err_msg = f"{action!r} ({type(action)}) invalid"
        assert self.action_space.contains(action), err_msg
        assert self.state is not None#, "Call reset before using step method."
        #x, theta = self.state # x, x_dot, theta, theta_dot
        
        
        print("action taken:", action)
        
        if action == 1:
                print("action taken:1")
        elif action == 0: 
                print("action taken:0")
        else:
                print("action taken:2")

        
        time.sleep(3)
        
        # wrong place to call them!
        # Sensor1 = board.get_pin('a:0:i')
        # Sensor2 = board.get_pin('a:2:i')
        
        Sensor1_messurment = Sensor1#humidity
        Sensor2_messurment = Sensor2#light
        
        
        print('Sensor1: Humidity state', Sensor1_messurment) 
        print('Sensor2: Light state', Sensor2_messurment) 
        
   
        time.sleep(2)
        #what is part of self-state aka world the agent observes
        Sensor1_messurment, Sensor2_messurment = self.state 

        
        #the sensor Agent should learn to not care about
        humidity_state = Sensor1
        #print('Humidity state', humidity_state)
        

        # the minus increment should be lower than the reward threshold
        battery_state =  Sensor2 -0.7  
        print('Battery state', battery_state)
        
        #time.sleep(10)
        
        self.state = (battery_state, humidity_state) 
        print('Self state', self.state)
        
        reward_battery = Sensor2
        print('Reward meassurment', reward_battery)
        
        if reward_battery > 0.75: 
            reward =1 
            print('1 reward')
            time.sleep(1)
        else: 
            reward = -1
            print('0 reward')
            time.sleep(1)

        #Are we done yet?
        if battery_state <= 0: 
                   done = True
        else:
                   done = False
                   
        time.sleep(2)
        print('---')
        info = {}


        # elif self.steps_beyond_done is None:
        #
        #     self.steps_beyond_done = 0
        #     reward = 1.0
        #     print('1 reward')
        #
        # # elif not done and Sensor2_messurment > 0.3:
        # #     reward = -2.0
        # #     print('-2 not reward')
        #
        #
        # else:
        #     if self.steps_beyond_done == 0:
        #         logger.warn(
        #             "You are calling 'step()' even though this "
        #             "environment has already returned done = True. You "
        #             "should always call 'reset()' once you receive 'done = "
        #             "True' -- any further steps are undefined behavior."
        #         )
        #     self.steps_beyond_done += 1
        #     reward = 0.0
        #     print('0 reward')

            

        return np.array(self.state, dtype=np.float32), reward, done, {}

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed)
        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(2,))
        self.steps_beyond_done = None
        if not return_info:
            return np.array(self.state, dtype=np.float32)
        else:
            return np.array(self.state, dtype=np.float32), {}

    def render(self, mode="human"):
    	pass
       
    def close(self):
    	pass