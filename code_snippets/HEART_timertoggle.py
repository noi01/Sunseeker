import time

#https://pyquesthub.com/creating-a-simple-stopwatch-in-python

countx = 0


class Heartpump:
    def __init__(self):
        self.start_time = 0
        self.elapsed_time = 0
        self.heartWaterLevelHigh = False

    def start(self):
        if not self.heartWaterLevelHigh:
            self.start_time = time.time() - self.elapsed_time
            self.heartWaterLevelHigh = True

    def stop(self):
        if self.heartWaterLevelHigh:
            self.elapsed_time = time.time() - self.start_time
            self.heartWaterLevelHigh = False

    def reset(self):
        self.start_time = 0
        self.elapsed_time = 0
        self.heartWaterLevelHigh = False

    def get_elapsed_time(self):
        if self.heartWaterLevelHigh:
            return time.time() - self.start_time
        return self.elapsed_time

# Example usage
sw = Heartpump()

for countx in range(10):
    if countx == 0:
        sw.start() 
        time.sleep(1)
        countx += 1
    elif countx == 9:
        sw.stop() 
        print(f'Elapsed time: {sw.get_elapsed_time()} seconds')  # should print approximately 2.0 seconds
        countx += 1
    else :
        countx += 1
        
    
'''
sw.start()  # start the stopwatch

time.sleep(2)  # wait for 2 seconds

sw.stop()  # stop the stopwatch
print(f'Elapsed time: {sw.get_elapsed_time()} seconds')  # should print approximately 2.0 seconds

#sw.reset()  # reset the stopwatch
'''