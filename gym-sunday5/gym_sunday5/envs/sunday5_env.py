import json
import threading
from typing import Optional, Union
import numpy as np
import gym
from gym import logger, spaces
from gym.error import DependencyNotInstalled
import time
import board

from misbkit import MisBKit

DEFAULT_WS_URL = "ws://192.168.0.125/ws"

ina219 = None
Second_Sensor = None
# if board is not connected use fake INA219 sensor
if board.board_id == "GENERIC_LINUX_PC":
    print("USING FAKE SENSORS")

    class INA219:
        def __init__(self, rng):
            self.rng = rng
            self.bus_voltage_range = 0x00
            self.gain = 0x00
            self.bus_adc_resolution = 0x00
            self.shunt_adc_resolution = 0x00
            self.mode = 0x05

            # Solar panel simulation parameters
            self._max_voltage = 18.0
            self._max_current = 1000.0  # in mA
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
            noise =  self.rng.uniform(-50, 50)  # ±50mA noise
            return max(0.0, base_current + noise)

        @property
        def voltage(self) -> float:
            """The bus voltage in volts."""
            base_voltage = self._max_voltage * (0.7 + 0.3 * self._output_level)
            noise =  self.rng.uniform(-0.1, 0.1)  # ±0.1V noise
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
            power_mw = self.voltage * self.current  # V * mA = mW
            return int(power_mw / self._power_lsb)

        @property
        def shunt_voltage(self) -> float:
            """Shunt voltage in millivolts."""
            return self.current * 0.1  # Assuming 0.1Ω shunt resistor

    

    class DigitalInOut:
        def __init__(self,rng):
            self.rng=rng
            pass

        @property
        def value(self):
            """The Fake Digital Pin Value"""
            return self.rng.choice([0, 1])

else:
    import digitalio
    from adafruit_ina219 import ADCResolution, BusVoltageRange, INA219

    # Second_Sensor = digitalio.DigitalInOut(board.D7)

    # Second_Sensor.direction = digitalio.Direction.INPUT

    # i2c_bus = board.I2C()  # uses board.SCL and board.SDA

    # ina219 = INA219(i2c_bus)


### Deleted everything from the motor instructions outside of print, because it's not important for testing / treat as placeholder
# Motor 1
def motorOFF():
    print(" motor01 OFF")

def motorCW():
    print(" motor01 CW")

def motorCWW():
    print(" motor01 CWW")


# Motor 2
def motor2OFF():
    print(" motor02 OFF")

def motor2CW():
    print(" motor02 CW")

def motor2CWW():
    print(" motor02 CWW")


# Motor 3
def motor3OFF():
    print(" motor03 OFF")

def motor3CW():
    print(" motor03 CW")

def motor3CWW():
    print(" motor03 CCW")





# `### Description

# This environment is for real time learning of hardware based agent. 

# For this example implementation the objective (goal) of the agent is to gather solar energy by turning solar panel towards the sun/light an increasing the charge of it's battery. At agents disposal are inputs from two sensors, the solar panel acting as a light sensor(Sensor 1), and second sensor that does not play part in energy gathering(Sensor 2).
# There is also third sensor (Sensor 3) - pertaining to charge of battery - that does not play part in the observation space, but only in reward calculation.


# ### Observation Space

# The observation is a `ndarray` with shape `(2,)` with the values


# | Num | Observation           | Min                 | Max               |
# |-----|-----------------------|---------------------|-------------------|
# | 0   | Sensor 1: Solar Panel | to be filed         | to be filed       |
# | 1   | Sensor 2              | to be filed         | to be filed       |


# ### Action Space

# The action is a discreet space of 9 corresponding to 3 predetermined joint position for each of of 3 legs.


# ### Rewards


# The reward is calculated by comparing the current battery charge (Sensor 3) with battery charge at previous time step. The reward is (+1) for battery charge increasing, and negative reward (-1) for battery charge decreasing. The reward is calculated at every step.


# ### Starting State

# To be filled in


# ### Episode End

# The episode ends if any one of the following occurs:

# Termination: Battery charge depletes to 0

# Truncation: To be filled in


# ### Learning loop structure (Step description)

# 1. The agent receives observation from Sensor 1 and Sensor 2

# 2. The agent choses an action from the discreete action space of 9

# 3. The rewards is calculated by comparing Sensor 3 at current time step to Sensor 3 at previous time step. If battery reaches 0 - the learning ends.


class sunday5(gym.Env[np.ndarray, Union[int, np.ndarray]]):

    def __init__(self):
        super().__init__()

        self.mbk = MisBKit()
        self.mbk._handle_sensor_data = self._parse_received_data
        self.mbk.connect()
        self._sensor_poll_stop = threading.Event()
        self._sensor_poll_thread = threading.Thread(
            target=self._poll_sensor_data,
            daemon=True,
        )
        self._sensor_poll_thread.start()
        
        self.JointPosition = 0
        self.Joint2Position = 0
        self.Joint3Position = 0

        #delaring sensors as part of the environment
        if board.board_id == "GENERIC_LINUX_PC":
            self.ina219 = INA219(self.np_random)
            self.dummy_sensor = DigitalInOut(self.np_random)
        else:
            self.dummy_sensor = digitalio.DigitalInOut(board.D7)
            self.dummy_sensor.direction = digitalio.Direction.INPUT
            self.ina219 = INA219(board.I2C())


        #why is this 10bit integer value if sensor data is float?
        self.Sensor_2_max = 1023  # INA219
        self.Sensor_2_min = 0

        self.Sensor_1_max = 1023  # Other sensor
        self.Sensor_1_min = 0

        self.sensors_data = np.array([0,0],dtype=np.float32)

        self.last_ina219 = {
            "current" : 0,
            "voltage" : 0,
            "shunt_voltage" : 0,
            "power" : 0
        
        }
        self.current_ina219 = {
            "current" : 0,
            "voltage" : 0,
            "shunt_voltage" : 0,
            "power" : 0
        }

        self.last_max17048 = {
            "cell_voltage" : 0,
            "percent" : 0,
            "rate":0
        }

        self.current_max17048 = {
            "cell_voltage" : 3.7,
            "percent" : 50,
            "rate":0
        }
        # define what actions are available to the agent
        self.action_space = spaces.Discrete(9)

        #define env observations
        self.observation_space = spaces.Box(
            low=np.array(
                [
                    self.Sensor_2_min, #is it important that sensor2 is first?
                    self.Sensor_1_min,
                ],
                dtype=np.float32,
            ),
            high=np.array(
                [
                    self.Sensor_2_max,
                    self.Sensor_1_max,
                ],
                dtype=np.float32,
            ),
            dtype=np.float32,
        )


        self.step_count = 0

        
            

    
    
    def _parse_received_data(self, val):
        print("popa")
        i2c_port = val["ports"][3]
        ina = None
        max17048 = None
        for u in i2c_port["units"]:
            if u["name"] == "sensor_ina219":
                ina = u
            elif u["name"] == "sensor_max17048":
                max17048 = u

        self.last_ina219 = self.current_ina219
        self.last_max17048 = self.current_max17048

        self.current_ina219["current"] = ina["val"][0]
        self.current_ina219["voltage"] = ina["val"][1]
        self.current_ina219["shunt_voltage"] = ina["val"][2]
        self.current_ina219["power"] = ina["val"][3]
        self.current_max17048["cell_voltage"] = max17048["val"][0]
        self.current_max17048["percent"] = max17048["val"][1]
        self.current_max17048["rate"] = max17048["val"][2]


   
    def _poll_sensor_data(self):
        while not self._sensor_poll_stop.wait(2.0):
            self.mbk.request_sensor_data()

    def _get_obs(self):
        """
        Get the current observation from the environment.

        This function samples the sensor data and stores it in `self.sensors_data`.
        The data is then clipped to be within the bounds of `self.observation_space`.
        The clipped data is then returned as the observation.

        Returns:
            np.ndarray: The observation from the environment.
        """
        # it seemed like the second sensor was never updated in the original code so i'm only
        # updating the current sensor. If other sensors are to be sampled add it here
        
        self.sensors_data[0] = np.float32(self.current_ina219["current"])  # Solar_panel
        self.sensors_data[1] = np.float32(self.current_max17048["cell_voltage"])
        self.sensors_data = np.clip(self.sensors_data, self.observation_space.low, self.observation_space.high)

        print("Observation")
        print(self.last_ina219)
        print(self.last_max17048)
        return np.array(self.sensors_data, dtype=np.float32)

    def _get_info(self):  
        """
        Retrieve the current state information of the robot.

        Returns:
            dict: A dictionary containing useful information about the env. 
        """

        return {
            "joint_pos": self.JointPosition,
            "joint2_pos": self.Joint2Position,
            "joint3_pos": self.Joint3Position,
            "step_count": self.step_count,
        }

    def _move_robot(self, action):
        """
        Move the robot's joints based on the given action.

        The function adjusts the positions of the robot's actuators according to the specified action.
        Each action corresponds to a specific movement of the joints.
        Args:
            action (int): The action to perform
        """
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
                motor3CWW()
                
            elif self.Joint3Position == 2:
                self.Joint3Position -= 2
                print("2-2 = joint3 at pos 0")
                motor3CWW()
                motor3CWW()
                
            else:
                print("Action error: action out of bounds")

        elif action == 7: #Motor3 pos 1
            if self.Joint3Position == 0:
                self.Joint3Position += 1
                print("0+1 = joint3 at pos 1")
                motor3CW()
                
            elif self.Joint3Position == 1:
                self.Joint3Position += 0
                print("joint3 at pos 1")
                
            elif self.Joint3Position == 2:
                self.Joint3Position -= 1
                print("joint3 at pos 2")
                motor3CWW()

            else:
                print("Action error: action out of bounds")

        elif action == 8: #Motor3 pos 2
            if self.Joint3Position == 0:
                self.Joint3Position += 2
                print("0+2 = joint3 at pos 2")
                motor3CW()
                motor3CW()
                
            elif self.Joint3Position == 1:
                self.Joint3Position += 1
                print("1+1 = joint3 at pos 2")
                motor3CW()
                
            elif self.Joint3Position == 2:
                self.Joint3Position += 0
                print("joint3 at pos 2")
                
            else:
                print("Action error: action out of bounds")

    def reset(self, seed: Optional[int] = None,options: Optional[dict] = None,):
        """
        Resets the environment to its initial state and returns the initial observation and info.

        Args:
            seed (Optional[int]): A seed for the random number generator.
            options (Optional[dict]): Additional options for the reset.

        Returns:
            tuple: A tuple containing the initial observation and information dictionary.
        """

        super().reset(seed=seed)
        
        if board.board_id == "GENERIC_LINUX_PC":
            self.ina219 = INA219(self.np_random)
            self.dummy_sensor = DigitalInOut(self.np_random)

        self.step_count = 0
      
        obs = self._get_obs()
        info = self._get_info()
        return obs, info
    
    def calculate_reward(self):
        reward = 0
        if self.current_max17048["cell_voltage"] > self.last_max17048["cell_voltage"]:
            reward = 1
        elif self.current_max17048["cell_voltage"] < self.last_max17048["cell_voltage"]:
            reward = -1

        return reward
        
    def step(self, action):  
        """
        Executes a single step in the environment using the given action.

        This function updates the environment's state based on the provided action,
        calculates the reward, and determines whether the episode has terminated.

        Args:
            action (int): An integer representing the action to be executed.
                        The action must be within the defined action space.

        Returns:
            tuple: A tuple containing the following elements:
                - observation (np.ndarray): The current observation of the environment.
                - reward (float): The reward obtained from the action.
                - terminated (bool): Whether the episode has terminated.
                - truncated (bool): Whether the episode was truncated.
                - info (dict): Additional information about the environment.
        """

        err_msg = f"{action!r} ({type(action)}) invalid"
        assert self.action_space.contains(action), err_msg

        self.step_count += 1
        truncated = False


        #update environment based on taken action. 
        #program environment logic here
        self._move_robot(action)
        observation = self._get_obs()

        #update current battery reading 

        if self.current_max17048["cell_voltage"] <= 3.0:
            terminated = True
        else:
            terminated = False

        reward =  self.calculate_reward()
        
        #end of step
        time.sleep(2)
        print("Action: {}, Voltage: {:.3f}, Reward: {}".format(action, self.current_max17048["cell_voltage"], reward))
        
        info = self._get_info()

        return observation, reward, terminated, truncated, info

    def close(self):
        self._sensor_poll_stop.set()
        if hasattr(self, "_sensor_poll_thread") and self._sensor_poll_thread.is_alive():
            self._sensor_poll_thread.join(timeout=2.0)
        if hasattr(self, "mbk"):
            self.mbk.close()
