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

'''
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
'''

    

### Classes for Misbikit



### Sensors

'''
    class DigitalInOut:
        def __init__(self,rng):
            self.rng=rng
            pass

        @property
        def value(self):
            """The Fake Digital Pin Value"""
            return self.rng.choice([0, 1])
            
 '''           

            



### !! ### For Etienne, I'm not sure if I need a seperate class for each sensor even if they are the same

#Sensor for Water Organ 1-3 are a Water Levele sensor (Grove / https://wiki.seeedstudio.com/Grove-Water-Level-Sensor/) placed within each hanging water vessel.

    class WaterOrgan1:
        def __init__(self,rng):
            self.rng=rng
            pass

        @property
        def value(self):
            """The Fake Digital Pin Value"""
            return self.rng.choice([0, 10])  
            
    class WaterOrgan2:
        def __init__(self,rng):
            self.rng=rng
            pass

        @property
        def value(self):
            """The Fake Digital Pin Value"""
            return self.rng.choice([0, 10])
    
    class WaterOrgan3:
        def __init__(self,rng):
            self.rng=rng
            pass

        @property
        def value(self):
            """The Fake Digital Pin Value"""
            return self.rng.choice([0, 10])
            
            

# Heart pump vessel water level
       
    class HeartPumpWaterSensor:
        def __init__(self, rng=None):
            self.rng=random.randint(1,100)
            pass

        @property
        def value(self):
            """The Fake Digital Pin Value"""
### ! ### For Etienne :      return self.rng.choice([0, 10]) was not working for me for passing between classes, maybe I messed something up so used this   
            rngValue = self.rng 
            return rngValue 






# Boleean if the hardware drowned or not

    class AquariumWaterLevel:
        def __init__(self,rng):
            self.rng=rng
            pass

        @property
        def value(self):
            """The Fake Digital Pin Value"""
            return self.rng.choice([0, 1])
            



### Physicial Motor Controls

            
### !! ###  For Etienne, should I make this into class
### Water solenoid control

    # Opens solenoid valve 1 for certain duration
    def solenoid_1_ON():
        print("solenoid 1")

    # Opens solenoid valve 2 for certain duration
    def solenoid_2_ON():
        print("solenoid 2")

    # Opens solenoid valve 3 for certain duration
    def solenoid_3_ON():
        print("solenoid 3")

    #  Opens the fourth valve that feeds back into main tank if none of soloneid valves were selected
    def solenoid_NONE():
        print("solenoid 4")        
            
            

    
            
            

            

else:
    import digitalio
    from adafruit_ina219 import ADCResolution, BusVoltageRange, INA219

    # Second_Sensor = digitalio.DigitalInOut(board.D7)



### This is a stopwatch class for counting the time elapsed between when the "heart" bladder is empty and full

class Heartpump:
    def __init__(self):
        self.start_time = 0
        self.elapsed_time = 0
        self.heartWaterLevelHigh = False

    def start(self):        #Heartpump.start()
        if not self.heartWaterLevelHigh:
            self.start_time = time.time() - self.elapsed_time
            self.heartWaterLevelHigh = True

    def stop(self):        #Heartpump.stop()
        if self.heartWaterLevelHigh:
            self.elapsed_time = time.time() - self.start_time
            self.heartWaterLevelHigh = False

    def reset(self):       #Heartpump.reset()
        self.start_time = 0
        self.elapsed_time = 0
        self.heartWaterLevelHigh = False

    def get_elapsed_time(self):     #Heartpump.get_elapsed_time()
        if self.heartWaterLevelHigh:
            return time.time() - self.start_time
        return self.elapsed_time





class HeartPumpControl:
   def process_data(self):
       hpws = HeartPumpWaterSensor()
       result = hpws.value
       result2 = hpws.value -10
       if result > 10:
           print(f"Pump On, water level: {result}")
               #pump on
       else :
               #pump off
           print(f"Pump OFF, water level: {result}")

     #  print(f"Processed: {result}")
      # print(f"Processed: {result2}")




'''
`### Description

This environment is for real time learning of hardware based agent.

The objective (goal) of the agent is to continuosly manage water level bewteen the main water tank and placed within it water organs,
so that the water level does not reach the hardware enclosure place in the middle of the main tank.


### Observation Space

The observation is a `ndarray` with shape `(2,)` with the values


| Num | Observation                           | Min                 | Max               |
|-----|---------------------------------------|---------------------|-------------------|
| 0   | Water Organ 1                         | to be filed         | to be filed       |
| 1   | Water Organ 2                         | to be filed         | to be filed       |
| 2   | Water Organ 3                         | to be filed         | to be filed       |
| 3   | Heartbeatpump.get_elapsed_time        | to be filed         | to be filed       |


Sensor for Water Organ 1-3 are a Water Levele sensor (Grove / https://wiki.seeedstudio.com/Grove-Water-Level-Sensor/) placed within each hanging water vessel.

Heartbeatpump.get_elapsed_time is an observation that meassures time elapsed between when heart water bladder going from empty to full. It uses Heartpump class.


### Action Space

The actions available to the agents is opening each of the three solenoid valves. The each valve connects the heart recepticle with one of the water organs.

The action is a discreet space of 8, corresponding to:

| Action number | Action                                                                          |
|---------------|---------------------------------------------------------------------------------|
| 0             | No water passed to Water Organs, water goes straigh back to recepticle          |
| 1             | Water passed to Water Organ 1                                                   |
| 2             | Water passed to Water Organ 2                                                   |
| 3             | Water passed to Water Organ 3                                                   |
| 4             | Water passed to Water Organ 1 & 2                                               |
| 5             | Water passed to Water Organ 1 & 3                                               |
| 6             | Water passed to Water Organ 2 & 3                                               |
| 7             | Water passed to Water Organ 1, 2 & 3                                            |

There is no specific amount of water passsed to each recepticle, each action corresponds to solenoid being open for a certain set duration of time while the pump is active.


### Rewards

The reward is a binary, taking agent_drowned sensor:
    If FALSE - the water did not reach hardware enclosure place and agent survived the cycle, the agent rewarded = 1. 
    If TRUE - the agent drowned, episode restarts, the agent rewarded = 0.

Based on Based on Yoshida, N. (2017). Homeostatic Agent for General Environment. Journal of Artificial General Intelligence p.4, 
Where reward (1) is equated to alive flag (1) vs dead flag (0) on Agent state.


### Starting State

To be filled in @_@


### Episode End

The episode ends if any one of the following occurs:

Agent drowned itself:
    agent_drowned sensor = TRUE

Truncation: To be filled in


### Learning loop structure (Step description)

1. The step starts when the heart water vessel is empty

2. The agent receives observation from the water organs 1-3 and time elapsed between heart water vessel going from empty to full

3. Heart water vessel becaming full automatically starts the water pump, which triggers action choice

4. The agent choses an action from the discreete action space of 8 on how to distrubute water



'''



class sunday5(gym.Env[np.ndarray, Union[int, np.ndarray]]):

    def __init__(self):
        super().__init__()


        #Misbikit setup
        
        self.mbk = MisBKit()
        self.mbk._handle_sensor_data = self._parse_received_data
        self.mbk.connect()
        self._sensor_poll_stop = threading.Event()
        self._sensor_poll_thread = threading.Thread(
            target=self._poll_sensor_data,
            daemon=True,
        )
        self._sensor_poll_thread.start()
        
 

        #delaring sensors as part of the environment
        if board.board_id == "GENERIC_LINUX_PC":
          
        #Added 
        
        #Sensor accessible to the agent:  
        
            self.waterORGANobs_1 = WaterOrgan1(self.np_random)
            self.waterORGANobs_2 = WaterOrgan2(self.np_random)
            self.waterORGANobs_3 = WaterOrgan3(self.np_random)

        
        #Sensor not for accessible to the agent:
        
        #Sensor if the heart recepticle is full
            self.heart_full_sensor = HeartPumpWaterLvL(self.np_random)
            
            
        #Sensor that the agent 'drowned'    
            self.agent_drowned = AquariumWaterLevel(self.np_random)
            
            
        else:
            self.dummy_sensor = digitalio.DigitalInOut(board.D7)
            self.dummy_sensor.direction = digitalio.Direction.INPUT
            self.ina219 = INA219(board.I2C())




        #why is this 10bit integer value if sensor data is float?
        ## N: No idea
        
        self.Sensor_1_max = 1023  # waterORGANobs_1
        self.Sensor_1_min = 0

        self.Sensor_2_max = 1023  # waterORGANobs_2
        self.Sensor_2_min = 0
        
        self.Sensor_3_max = 1023  # waterORGANobs_3
        self.Sensor_3_min = 0
        
        self.Sensor_4_max = 1023  # heart_full_sensor
        self.Sensor_4_min = 0
        

        
        

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
        self.action_space = spaces.Discrete(8)

        #define env observations
        self.observation_space = spaces.Box(
            low=np.array(
                [
                    self.Sensor_1_min, 
                    self.Sensor_2_min,
                    self.Sensor_3_min,
                    self.Sensor_4_min,
                    
                ],
                dtype=np.float32,
            ),
            high=np.array(
                [
                    self.Sensor_1_max,
                    self.Sensor_2_max,
                    self.Sensor_3_max,
                    self.Sensor_4_max,
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







###### DONE #####
    def _move_robot(self, action):
        """
        
        Solenoid actions
        
        Args:
            action (int): The action to perform
        """

## No solenoid open - water goes straight back to receptacle

        if action == 0: 
            solenoid_NONE():
            else:
                print("Action error: action out of bounds")

## Single valve open actions

        elif action == 1:
            solenoid_1_ON()        
            else:
                print("Action error: action out of bounds")

        elif action == 2:
            solenoid_2_ON()
            else:
                print("Action error: action out of bounds")

        elif action == 3: 
            solenoid_3_ON()  
            else:
                print("Action error: action out of bounds")

## Double valves open action

        elif action == 4: 
            solenoid_1_ON()
            solenoid_2_ON()
            else:
                print("Action error: action out of bounds")

        elif action == 5: #Motor2 pos 2
            solenoid_1_ON()
            solenoid_3_ON()
            else:
                print("Action error: action out of bounds")

        elif action == 6: #Motor3 pos 0
            solenoid_2_ON()
            solenoid_3_ON()
            else:
                print("Action error: action out of bounds")

## Triple valves open action

        elif action == 7: #Motor3 pos 1
            solenoid_1_ON()
            solenoid_2_ON()
            solenoid_3_ON()
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
        
        # Based on Yoshida, N. (2017). Homeostatic Agent for General Environment. Journal of Artificial General Intelligence p.4
        # The alive (1) vs dead (0) Agent state 

        
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
