import math
from typing import Optional, Union

import numpy as np
import random

import gym
from gym import logger, spaces
from gym.error import DependencyNotInstalled

import time

## Circuitpython

import board
import digitalio

from adafruit_ina219 import ADCResolution, BusVoltageRange, INA219

Second_Sensor = digitalio.DigitalInOut(board.D7)

Second_Sensor.direction = digitalio.Direction.INPUT

i2c_bus = board.I2C()  # uses board.SCL and board.SDA

ina219 = INA219(i2c_bus)

print("ina219 test")


## display some of the advanced field (just to test)
print("Config register:")
print("  bus_voltage_range:    0x%1X" % ina219.bus_voltage_range)
print("  gain:                 0x%1X" % ina219.gain)
print("  bus_adc_resolution:   0x%1X" % ina219.bus_adc_resolution)
print("  shunt_adc_resolution: 0x%1X" % ina219.shunt_adc_resolution)
print("  mode:                 0x%1X" % ina219.mode)
print("")

Value_Second_Sensor = Second_Sensor.value

print(Value_Second_Sensor)


### Deleted everything from the motor instructions outside of print, because it's not important for testing / treat as placeholder

#Motor 1

def motorOFF():
        print(" motor01 OFF")
        
def motorCW():
        print(" motor01 CW")
        
def motorCWW():
        print(" motor01 CWW")

#Motor 2

def motor2OFF():
        print(" motor02 OFF")
        
def motor2CW():
        print(" motor02 CW")
        
def motor2CWW():
        print(" motor02 CWW")

#Motor 3

def motor3OFF():
        print(" motor03 OFF")
        
def motor3CW():
        print(" motor03 CW")
        
def motor3CWW():
        print(" motor03 CCW")

JointPosition = 0

Joint2Position = 0

Joint3Position = 0


class sunday5(gym.Env[np.ndarray, Union[int, np.ndarray]]):


    def __init__(self):
        
        self.battery = 10 #uh.... forgot what this does
        
        self.Sensor_2_max = 1023 #INA219
        self.Sensor_2_min = 0
        
        self.Sensor_1_max = 1023 #Other sensor
        self.Sensor_1_min = 0

        # Angle limit set to 2 * theta_threshold_radians so failing observation
        # is still within bounds.
        high = np.array(
            [
                self.Sensor_2_max,
                self.Sensor_1_max,
 
            ],
            dtype=np.float32,
        )
        
        low = np.array(
            [
                self.Sensor_2_min,
                self.Sensor_1_min,
 
            ],
            dtype=np.float32,
        )


        self.action_space = spaces.Discrete(9) 
        self.observation_space = spaces.Box(low, high, dtype=np.float32)


        self.state = None

        self.steps_beyond_done = None

    def step(self, action):
        # actuation, it needs it to work
        global JointPosition
        global Joint2Position
        global Joint3Position

        err_msg = f"{action!r} ({type(action)}) invalid"
        assert self.action_space.contains(action), err_msg
        assert self.state is not None #, "Call reset before using step method."

        
        
        print("Action taken:", action)

        
### What actions do is not important, treat as placeholder
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

        elif action == 3: #Motor2 pos 0
            if Joint2Position == 0:
                Joint2Position += 0
                print("joint2 at pos 0")

                
            elif Joint2Position == 1:
                Joint2Position -= 1
                print("0-1 = joint2 at pos 0")
                motor2CWW()
                
            elif Joint2Position == 2:
                Joint2Position -= 2
                print("2-2 = joint2 at pos 0")
                motor2CWW()
                motor2CWW()
                
            else:
                print("Action error: action out of bounds")

        elif action == 4: #Motor2 pos 1
            if Joint2Position == 0:
                Joint2Position += 1
                print("0+1 = joint2 at pos 1")
                motor2CW()
                
            elif Joint2Position == 1:
                Joint2Position += 0
                print("joint2 at pos 1")
                
            elif Joint2Position == 2:
                Joint2Position -= 1
                print("joint2 at pos 2")
                motor2CWW()

            else:
                print("Action error: action out of bounds")

        elif action == 5: #Motor2 pos 2
            if Joint2Position == 0:
                Joint2Position += 2
                print("0+2 = joint2 at pos 2")
                motor2CW()
                motor2CW()
                
            elif Joint2Position == 1:
                Joint2Position += 1
                print("1+1 = joint2 at pos 2")
                motor2CW()
                
            elif Joint2Position == 2:
                Joint2Position += 0
                print("joint2 at pos 2")

            else:
                print("Action error: action out of bounds")

        elif action == 6: #Motor3 pos 0
            if Joint3Position == 0:
                Joint3Position += 0
                print("joint3 at pos 0")

                
            elif Joint3Position == 1:
                Joint3Position -= 1
                print("0-1 = joint3 at pos 0")
                motor2CWW()
                
            elif Joint3Position == 2:
                Joint3Position -= 2
                print("2-2 = joint3 at pos 0")
                motor2CWW()
                motor2CWW()
                
            else:
                print("Action error: action out of bounds")

        elif action == 7: #Motor3 pos 1
            if Joint3Position == 0:
                Joint3Position += 1
                print("0+1 = joint3 at pos 1")
                motor2CW()
                
            elif Joint3Position == 1:
                Joint3Position += 0
                print("joint3 at pos 1")
                
            elif Joint3Position == 2:
                Joint3Position -= 1
                print("joint3 at pos 2")
                motor2CWW()

            else:
                print("Action error: action out of bounds")

        elif action == 8: #Motor3 pos 2
            if Joint3Position == 0:
                Joint3Position += 2
                print("0+2 = joint3 at pos 2")
                motor2CW()
                motor2CW()
                
            elif Joint3Position == 1:
                Joint3Position += 1
                print("1+1 = joint3 at pos 2")
                motor2CW()
                
            elif Joint3Position == 2:
                Joint3Position += 0
                print("joint3 at pos 2")
                
            else:
                print("Action error: action out of bounds")

        print('Joint 1 position:', JointPosition,'Joint 2 position:', Joint2Position, 'Joint 3 position:', Joint3Position)
        
        
        
        
        Sensor1_messurment = Value_Second_Sensor #Second_Sensor
        Sensor2_messurment = ina219.current #Solar_panel
        
        
        print('Sensor1: Other Sensor', Sensor1_messurment) 
        print('Sensor2: INA219 current', Sensor2_messurment) 
        

        #what is part of self-state aka world the agent observes
        Sensor1_messurment, Sensor2_messurment = self.state 

        
        #the sensor Agent should learn to not care about
        Other_sensor_state = Sensor1_messurment
        

        # there will be diffrent math later for how much energy agent gathered, this is just a placeholder 
        INA219_state =  Sensor2_messurment -0.7  
        print('Battery state', INA219_state)
        
        
        
        #Env observation
        self.state = (INA219_state, Other_sensor_state) 
        print('Self state - INA219 current - Other Sensor', self.state)
        
        
        
        ## Reward
        # there will be diffrent math later for reward for now just so there is some sensor value, this is just a placeholder 
        reward_battery = Sensor2_messurment
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
        if INA219_state <= 0: 
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
