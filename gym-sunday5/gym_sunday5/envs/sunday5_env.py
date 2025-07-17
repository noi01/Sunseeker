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

ina219 = None
Second_Sensor = None
#if board is not connected use fake INA219 sensor

if board.board_id =="GENERIC_LINUX_PC":
    print("USING FAKE SENSORS")
    class INA219:
        def __init__(self):
            print("Fake INA219")
            self.bus_voltage_range = 0x00
            self.gain = 0x00
            self.bus_adc_resolution = 0x00
            self.shunt_adc_resolution = 0x00
            self.mode = 0x05
            
            # Solar panel simulation parameters
            self._max_voltage = 18.0
            self._max_current = 3000.0  # in mA
            self._output_level = 0.5  # 0.0 to 1.0
            
            # INA219 calibration values
            self._cal_value = 4096
            self._power_lsb = 2.0  # mW per bit
            self._raw_calibration = self._cal_value
        
        def set_output_level(self, level):
            """Set solar panel output level (0.0 to 1.0)"""
            self._output_level = max(0.0, min(1.0, level))
        
        @property
        def current(self) -> float:
            """The current through the shunt resistor in milliamps."""
            base_current = self._max_current * self._output_level
            noise = random.uniform(-50, 50)  # ±50mA noise
            return max(0.0, base_current + noise)
        
        @property
        def voltage(self) -> float:
            """The bus voltage in volts."""
            base_voltage = self._max_voltage * (0.7 + 0.3 * self._output_level)
            noise = random.uniform(-0.1, 0.1)  # ±0.1V noise
            return max(0.0, base_voltage + noise)
        
        @property
        def power(self) -> float:
            """The power through the load in Watt."""
            # Sometimes a sharp load will reset the INA219, which will
            # reset the cal register, meaning CURRENT and POWER will
            # not be available ... always setting a cal
            # value even if it's an unfortunate extra step
            self._raw_calibration = self._cal_value
            # Now we can safely read the CURRENT register!
            return self.raw_power * self._power_lsb / 1000.0  # Convert mW to W
        
        @property
        def raw_power(self) -> int:
            """Raw power register value."""
            power_mw = (self.voltage * self.current)  # V * mA = mW
            return int(power_mw / self._power_lsb)
        
        @property
        def shunt_voltage(self) -> float:
            """Shunt voltage in millivolts."""
            return self.current * 0.1  # Assuming 0.1Ω shunt resistor


    ina219 = INA219()

    class DigitalInOut:
        def __init__(self):
             pass
        @property
        def value(self):
            """The Fake Digital Pin Value"""
            return random.choice([0, 1])
        
    Second_Sensor = DigitalInOut()
else:
    import digitalio
    from adafruit_ina219 import ADCResolution, BusVoltageRange, INA219

    Second_Sensor = digitalio.DigitalInOut(board.D7)

    Second_Sensor.direction = digitalio.Direction.INPUT

    i2c_bus = board.I2C()  # uses board.SCL and board.SDA

    ina219 = INA219(i2c_bus)



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



class sunday5(gym.Env[np.ndarray, Union[int, np.ndarray]]):


    def __init__(self):
        
        self.battery = 10 #uh.... forgot what this does
        
        self.Sensor_2_max = 1023 #INA219
        self.Sensor_2_min = 0
        
        self.Sensor_1_max = 1023 #Other sensor
        self.Sensor_1_min = 0

        self.JointPosition = 0

        self.Joint2Position = 0

        self.Joint3Position = 0


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

        err_msg = f"{action!r} ({type(action)}) invalid"
        assert self.action_space.contains(action), err_msg
        assert self.state is not None #, "Call reset before using step method."

        
        
        print("Action taken:", action)

        
### What actions do is not important, treat as placeholder
        if action == 0: 
            if self.JointPosition == 0:
                self.JointPosition += 0
                print("joint at pos 0")
                
            elif self.JointPosition == 1:
                self.JointPosition -= 1
                print("1-1 = joint at pos 0")
                motorCWW()
                
            elif self.JointPosition == 2:
                self.JointPosition -= 2
                print("2-2 = joint at pos 0")
                motorCWW()
                motorCWW()
                
            else:
                print("Action error: action out of bounds")

        elif action == 1:
            if self.JointPosition == 0:
                self.JointPosition += 1
                print("0+1 = joint at pos 1")
                motorCW()
                
            elif self.JointPosition == 1:
                self.JointPosition += 0
                print("joint at pos 1")
                
            elif self.JointPosition == 2:
                self.JointPosition -= 1
                print("2-1 = joint at pos 1")
                motorCWW()
                
            else:
                print("Action error: action out of bounds")

        elif action == 2:
            if self.JointPosition == 0:
                self.JointPosition += 2
                print("0+2 = joint at pos 2")
                motorCW()
                motorCW()
                
            elif self.JointPosition == 1:
                self.JointPosition += 1
                print("1+1 = joint at pos 2")
                motorCW()
                
            elif self.JointPosition == 2:
                self.JointPosition += 0
                print("joint at pos 2")
                
            else:
                print("Action error: action out of bounds")

        elif action == 3: #Motor2 pos 0
            if self.Joint2Position == 0:
                self.Joint2Position += 0
                print("joint2 at pos 0")

                
            elif self.Joint2Position == 1:
                self.Joint2Position -= 1
                print("0-1 = joint2 at pos 0")
                motor2CWW()
                
            elif self.Joint2Position == 2:
                self.Joint2Position -= 2
                print("2-2 = joint2 at pos 0")
                motor2CWW()
                motor2CWW()
                
            else:
                print("Action error: action out of bounds")

        elif action == 4: #Motor2 pos 1
            if self.Joint2Position == 0:
                self.Joint2Position += 1
                print("0+1 = joint2 at pos 1")
                motor2CW()
                
            elif self.Joint2Position == 1:
                self.Joint2Position += 0
                print("joint2 at pos 1")
                
            elif self.Joint2Position == 2:
                self.Joint2Position -= 1
                print("joint2 at pos 2")
                motor2CWW()

            else:
                print("Action error: action out of bounds")

        elif action == 5: #Motor2 pos 2
            if self.Joint2Position == 0:
                self.Joint2Position += 2
                print("0+2 = joint2 at pos 2")
                motor2CW()
                motor2CW()
                
            elif self.Joint2Position == 1:
                self.Joint2Position += 1
                print("1+1 = joint2 at pos 2")
                motor2CW()
                
            elif self.Joint2Position == 2:
                self.Joint2Position += 0
                print("joint2 at pos 2")

            else:
                print("Action error: action out of bounds")

        elif action == 6: #Motor3 pos 0
            if self.Joint3Position == 0:
                self.Joint3Position += 0
                print("joint3 at pos 0")

                
            elif self.Joint3Position == 1:
                self.Joint3Position -= 1
                print("0-1 = joint3 at pos 0")
                motor2CWW()
                
            elif self.Joint3Position == 2:
                self.Joint3Position -= 2
                print("2-2 = joint3 at pos 0")
                motor2CWW()
                motor2CWW()
                
            else:
                print("Action error: action out of bounds")

        elif action == 7: #Motor3 pos 1
            if self.Joint3Position == 0:
                self.Joint3Position += 1
                print("0+1 = joint3 at pos 1")
                motor2CW()
                
            elif self.Joint3Position == 1:
                self.Joint3Position += 0
                print("joint3 at pos 1")
                
            elif self.Joint3Position == 2:
                self.Joint3Position -= 1
                print("joint3 at pos 2")
                motor2CWW()

            else:
                print("Action error: action out of bounds")

        elif action == 8: #Motor3 pos 2
            if self.Joint3Position == 0:
                self.Joint3Position += 2
                print("0+2 = joint3 at pos 2")
                motor2CW()
                motor2CW()
                
            elif self.Joint3Position == 1:
                self.Joint3Position += 1
                print("1+1 = joint3 at pos 2")
                motor2CW()
                
            elif self.Joint3Position == 2:
                self.Joint3Position += 0
                print("joint3 at pos 2")
                
            else:
                print("Action error: action out of bounds")

        print('Joint 1 position:', self.JointPosition,'Joint 2 position:', self.Joint2Position, 'Joint 3 position:', self.Joint3Position)
        
        
        
        
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
