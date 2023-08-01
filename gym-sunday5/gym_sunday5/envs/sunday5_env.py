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
port = '/dev/cu.usbmodem143101'
pin  = 9
board = Arduino(port)

iterator = util.Iterator(board)
iterator.start()

Sensor1 = board.get_pin('a:0:i')
Sensor2 = board.get_pin('a:1:i')

time.sleep(2)

DC_motor_pin3 = board.get_pin('d:3:o')
DC_motor_pin2 = board.get_pin('d:2:o')

def motorOFF():
        DC_motor_pin3.write(1)
        DC_motor_pin2.write(1)
        time.sleep(0.01)
        
def motorCW():
        DC_motor_pin3.write(1)
        DC_motor_pin2.write(1)
        time.sleep(0.01)
        
        DC_motor_pin3.write(1)
        DC_motor_pin2.write(0)
        
        time.sleep(1)
        
        DC_motor_pin3.write(1)
        DC_motor_pin2.write(1)
        
        time.sleep(1)
        
def motorCWW():
        DC_motor_pin3.write(1)
        DC_motor_pin2.write(1)
        time.sleep(0.01)
        
        DC_motor_pin3.write(0)
        DC_motor_pin2.write(1)
        
        time.sleep(1)
        
        DC_motor_pin3.write(1)
        DC_motor_pin2.write(1)
        
        time.sleep(1)

JointPosition = 0

motorCWW()
motorCWW()

class sunday5(gym.Env[np.ndarray, Union[int, np.ndarray]]):


    #metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(self):
        
        self.battery = 10
        
        
        self.solar_panel_max = 1023
        self.solar_panel_min = 0
        
        self.humidity_sensor_max = 1023
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


        self.state = None

        self.steps_beyond_done = None

    def step(self, action):
        # actuation
        global JointPosition

        err_msg = f"{action!r} ({type(action)}) invalid"
        assert self.action_space.contains(action), err_msg
        assert self.state is not None#, "Call reset before using step method."
        #x, theta = self.state # x, x_dot, theta, theta_dot
        
        
        print("action taken:", action)

        

        if action == 0:
            if JointPosition == 0:
                JointPosition += 0
                print("joint at pos 0")
                
            elif JointPosition == 1:
                JointPosition -= 1
                print("1-1 = joint at pos 0")
                motorCWW()
                
            elif JointPosition == 2:
                JointPosition -= 2
                print("2-2 = joint at pos 0")
                motorCWW()
                motorCWW()
                
            else:
                print("Action error: action out of bounds")

        elif action == 1:
            if JointPosition == 0:
                JointPosition += 1
                print("0+1 = joint at pos 1")
                motorCW()
                
            elif JointPosition == 1:
                JointPosition += 0
                print("joint at pos 1")
                
            elif JointPosition == 2:
                JointPosition -= 1
                print("2-1 = joint at pos 1")
                motorCWW()
                
            else:
                print("Action error: action out of bounds")
        elif action == 2:
            if JointPosition == 0:
                JointPosition += 2
                print("0+2 = joint at pos 2")
                motorCW()
                motorCW()
                
            elif JointPosition == 1:
                JointPosition += 1
                print("1+1 = joint at pos 2")
                motorCW()
                
            elif JointPosition == 2:
                JointPosition += 0
                print("joint at pos 2")
                
            else:
                print("Action error: action out of bounds")



        print('join position', JointPosition)
        
        # wrong place to call them!
        # Sensor1 = board.get_pin('a:0:i')
        # Sensor2 = board.get_pin('a:2:i')
        
        Sensor1_messurment = Sensor1.read()#humidity
        Sensor2_messurment = Sensor2.read()#light
        
        
        print('Sensor1: Humidity state', Sensor1_messurment) 
        print('Sensor2: Light state', Sensor2_messurment) 
        
   
        
        #what is part of self-state aka world the agent observes
        Sensor1_messurment, Sensor2_messurment = self.state 

        
        #the sensor Agent should learn to not care about
        humidity_state = Sensor1.read()
        #print('Humidity state', humidity_state)
        

        # the minus increment should be lower than the reward threshold
        battery_state =  Sensor2.read() -0.7  
        print('Battery state', battery_state)
        
        #time.sleep(10)
        
        self.state = (battery_state, humidity_state) 
        print('Self state - battery - humidity', self.state)
        
        reward_battery = Sensor2.read()
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