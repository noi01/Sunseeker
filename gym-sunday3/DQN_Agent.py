import numpy as np

import time


#Agent dependencies

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam

from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory


#Agent



actions = env.action_space.n  # dim of output layer 
input_dim = env.observation_space.shape[0]  # dim of input layer 




#Model of the DQN

def build_model(states, actions):

    model = Sequential()
    model.add(Dense(32, input_dim = input_dim , activation = 'relu'))
    model.add(Dense(64, activation = 'relu'))
    model.add(Dense(actions, activation = 'linear'))
    model.compile(optimizer=Adam(), loss = 'mse')
    return model
    
#del model 

model = build_model(input_dim, actions)

model.summary()

#Build the Agent -  combine DQN model and actions

def build_agent(model, n_actions):
    policy = BoltzmannQPolicy()
    memory = SequentialMemory(limit=50000, window_length=2)
    dqn = DQNAgent(model=model, memory=memory, policy=policy, 
                  nb_actions=actions, nb_steps_warmup=40, target_model_update=1e-2)
    return dqn
    
dqn = build_agent(model, actions)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])
dqn.fit(env, nb_steps=700, visualize=False, verbose=2)# 0 for no logging, 1 for interval logging (compare log_interval), 2 for episode logging


scores = dqn.test(env, nb_episodes=10, visualize=False)
print(np.mean(scores.history['episode_reward']))
