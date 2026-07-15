import numpy as np
import random

#Test on passing value from simulated Heart Water Sensor to Pump Control
       
class HeartPumpWaterSensor:
    def __init__(self, rng=None):
        self.rng=random.randint(1,100)
        pass

    @property
    def value(self):
        """The Fake Digital Pin Value"""
        
        rngValue = self.rng
        return rngValue 


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
       
       
b = HeartPumpControl()
b.process_data()