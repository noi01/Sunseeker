

# Version of libraries: Python 3.11.0 , numpy 2.0.2 , tensorflow 2.18.0 , keras 3.6.0,
# Hardware interfacing: Adafruit-Blinka 8.61.2 , adafruit-circuitpython-ina219 3.4.2

# gym 0.26.2 installed from  repo that fixes np.bool8 with : pip install git+https://github.com/sebastianbrzustowicz/gym.git@np.bool_
# from https://github.com/openai/gym/pull/3258


import random
import gym
import numpy as np
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

EPISODES = 100


class DqnAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


if __name__ == "__main__":
    
    #Environment 
    import gym_sunday5
    env = gym.make('sunday5-v1')
   
    output_file = open("sunday5-v1_output.csv","w+")
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DqnAgent(state_size, action_size)

    ## This should be tested so it saves / loads
    # agent.load("./save/sunday3-dqn.h5")

    done = False
    batch_size = 10
    count = 0

    for e in range(EPISODES):
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        for time in range(500):
            # env.render() #no preview of environment
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            reward = reward if not done else -10
            next_state = np.reshape(next_state, [1, state_size])
            agent.remember(state, action, reward, next_state, done)

            state = next_state

            if done:
                print("episode: {}/{}, score: {}, e: {:.2}"
                      .format(e, EPISODES, time, agent.epsilon))
                output = str(e) + ", " + str(time) + ", " + str(agent.epsilon) + "\n"
                output_file.write(output)
                output_file.flush()
                break
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)
                
            ## This should be tested so it saves / loads
            #if e % 10 == 0:
             #   agent.save("./save/sunday3-dqn.h5")
             
    output_file.close()
