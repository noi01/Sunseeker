import math
from typing import Optional

import numpy as np

import time

#Gym dependencies
import gym
from gym import error, spaces, utils, wrappers
from gym.utils import seeding

import gym_sunday1   
env_make = gym.make('sunday1-v1')

env = sunday1()  

#Agent dependencies
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam

from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory

from DQN_Agent import dqn, model, actions, build_agent

#Main script




actions = env.action_space.n  # dim of output layer 
input_dim = env.observation_space.shape[0] 

#print(env.observation_space.sample())

episodes = 1
for episode in range(1, episodes+1):
    state = env.reset()
    done = False
    score = 0 
    
    while not done:
        #env.render()
        action = env.action_space.sample()
        n_state, reward, done, info = env.step(action)
        score+=reward
    print('Episode:{} Score:{}'.format(episode, score))     
  

# actions = env.action_space.n  # dim of output layer
# input_dim = env.observation_space.shape[0]  # dim of input layer 

    
dqn = build_agent(model, actions)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])
dqn.fit(env, nb_steps=700, visualize=False, verbose=2)# 0 for no logging, 1 for interval logging (compare log_interval), 2 for episode logging


scores = dqn.test(env, nb_episodes=10, visualize=False)
print(np.mean(scores.history['episode_reward']))
