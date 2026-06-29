

# Version of libraries: Python 3.11.0 , numpy 2.0.2 , tensorflow 2.18.0 , keras 3.6.0,
# Hardware interfacing: Adafruit-Blinka 8.61.2 , adafruit-circuitpython-ina219 3.4.2

# gym 0.26.2 installed from  repo that fixes np.bool8 with : pip install git+https://github.com/sebastianbrzustowicz/gym.git@np.bool_
# from https://github.com/openai/gym/pull/3258

import os
import random
import datetime
import argparse
import gym
import numpy as np
from collections import deque
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import Callback
import gym_sunday5


#############################################################################
#                                 Constant                                  #
#############################################################################
EPISODES = 100
MAXIMUM_STEPS = 50
BATCH_SIZE = 10
SAVE_INTERVAL = 10
EXPERIMENT_PREFIX = "sunseeker"
FOLDER_TO_SAVE_TO = "models"
WS_URL = "ws://192.168.0.125/ws"


#############################################################################
#                              Model definition                             #
#############################################################################
class GetWeights(Callback):
    # Keras callback which collects values of weights and biases at each epoch
    def __init__(self):
        super(GetWeights, self).__init__()
        self.weight_dict = {}

    def on_epoch_end(self, epoch, logs=None):
        # this function runs at the end of each epoch

        # loop over each layer and get weights and biases
        for layer_i in range(len(self.model.layers)):
            w = self.model.layers[layer_i].get_weights()[0]
            b = self.model.layers[layer_i].get_weights()[1]
            print('Layer %s has weights of shape %s and biases of shape %s' %(
                layer_i, np.shape(w), np.shape(b)))

            # save all weights and biases inside a dictionary
            if epoch == 0:
                # create array to hold weights and biases
                self.weight_dict['w_'+str(layer_i+1)] = w
                self.weight_dict['b_'+str(layer_i+1)] = b
            else:
                # append new weights to previously-created weights array
                self.weight_dict['w_'+str(layer_i+1)] = np.dstack(
                    (self.weight_dict['w_'+str(layer_i+1)], w))
                # append new weights to previously-created weights array
                self.weight_dict['b_'+str(layer_i+1)] = np.dstack(
                    (self.weight_dict['b_'+str(layer_i+1)], b))

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
        model.add(Input(shape = (self.state_size,)))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(learning_rate=self.learning_rate))
        print()
        model.summary(show_trainable=True)
        print()
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.randint(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action
    
    def printWeights(self):
        for i, layer in enumerate(self.model.layers):
            print(f"Layer {i}: {layer.name}")
            print(layer.get_weights())
            print()

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target

            h = self.model.fit(state, target_f, epochs=1, verbose=0)
        print(h.history)  
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
  
    def load(self, name):
        print("Loading model: {}".format(name))
        self.model = tf.keras.models.load_model(name)
        self.model.summary()
        print()
        self.printWeights()

    def save(self, name):
        self.model.save("{}.keras".format(name))
     
def verify_environment():
    """
    Verify the gym environment, by calling check_env from gym.utils.env_checker

    This function prints a success message if the environment is correct or stop the
    code if it fails
    """
    print("----------------------------- Checking environment ------------------------------------")
    from gym.utils.env_checker import check_env
    env = gym.make('sunday5-v1')
    check_env(env.unwrapped)
    print("----------------------------- env check successfull ------------------------------------")

def build_paths():
    """
    Generate paths for model saving and experiment tracking based on the script launch time

    :return: path_to_experiment_folder, path_to_model, path_to_output
    :rtype: tuple
    """
    launch_time = datetime.datetime.now().strftime('%y%m%d_%H%M')
    experiment_name = "{}_{}".format(EXPERIMENT_PREFIX,launch_time)
    agent_name = "agent_{}".format(launch_time)

    print("Experiment: {}".format(experiment_name))

    #generate paths for model saving
    path_to_save_folder = os.path.join(os.getcwd(),FOLDER_TO_SAVE_TO)
    path_to_experiment_folder = os.path.join(path_to_save_folder,experiment_name)
    path_to_model = os.path.join(path_to_experiment_folder, agent_name)
    path_to_output = os.path.join(path_to_experiment_folder, "_ouptut.csv")
    return path_to_experiment_folder, path_to_model, path_to_output


#############################################################################
#                              Script Launch                                #
#############################################################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c','--check_env', action="store_true", help="Test custom envrionment with the check_env function provider by gym before launching experiment")
    parser.add_argument('-e', '--episode', type=int,default=EPISODES, help="Number of episodes the agent will run for")
    parser.add_argument('-m', "--max_steps", type=int, default=MAXIMUM_STEPS, help="Maximum of steps the agent can take befre truncating the episode")
    parser.add_argument('-b', "--batch_size", type=int, default=BATCH_SIZE, help="Agent's batch size")
    parser.add_argument('-s', "--save_interval", type=int, default=SAVE_INTERVAL, help="Number of completed episode after which the agent will be saved")
    parser.add_argument('-l', "--load",type=str, help="Path to the .keras file of the model to load" )
    parser.add_argument('--ws_url', type=str, default=WS_URL, help="WebSocket server URL, ex: ws://localhost:8765")
    parser.add_argument('--ws_timeout', type=float, default=5.0, help="Timeout in seconds used while establishing websocket connection")

    args = parser.parse_args()
    if args.check_env == True:
        verify_environment()
    #########################################################################
    #                          Experiment setup                             #
    #########################################################################
    print("\n----------------------------- Experiment start ------------------------------------")

    print("\n\nSunseeker - Natalia Balska\n\n")
    path_to_experiment_folder,path_to_model, path_to_output = build_paths()
    
    os.makedirs(path_to_experiment_folder, exist_ok= True)
    output_file = open(path_to_output,"w+")

    print("Experiment folder setup")

    #Environment 
    base_env = gym.make('sunday5-v1', max_episode_steps = args.max_steps)
    env = gym.wrappers.RecordEpisodeStatistics(base_env)
    print("Environment setup")
    
 
    state_size = env.observation_space.shape[0] 
    action_size = env.action_space.n

    
    agent = DqnAgent(state_size, action_size)
    if args.load is not None:
        agent.load(args.load)


  
  


    # while True:
    #     time.sleep(0.01)
    try:
        print("----------------------------- Training loop start ------------------------------------")
        #########################################################################
        #                             Training loop                             #
        #########################################################################
        for e in range(args.episode):
            done = False #episode done
            episode_count = e+1 #add 1 to count episode starting at episode 1

            state, info = env.reset()
            state = np.reshape(state, [1, state_size])

            #timestep loop
            #it looks like some of this code should be written inside the env code instead of the training loop
            print("==============================================================")
            while not done:

                print("\nEPISODE: {} STEP: {}".format(episode_count,info["step_count"]))
                action = agent.act(state)
                observation, reward, done, truncated, info = env.step(action)
                print(info)
                if truncated:
                    print("Episode truncated")
                    #implement the truncation logic here
                    break

                #here implement the reward logic through episodes.
                reward = reward if not done else -10

                observation = np.reshape(observation, [1, state_size])
                agent.remember(state, action, reward, observation, done)

                state = observation

                if done:
                    #shouldnt score be  formated with reward instead of time?
                    print("Episode done")
                    print("Episode: {}/{}, score: {}, epsilon: {:.2}".format(episode_count, args.episode, info["step_count"], agent.epsilon))
                    # agent.printWeights()
                    #format ouptut and write to csv file
                    output = str(e) + ", " + str(info["step_count"]) + ", " + str(agent.epsilon) + "\n"
                    output_file.write(output)
                    output_file.flush()

                    # save model if interval is reached and this is not the last episode
                    if episode_count % args.save_interval == 0 and episode_count != args.episode:
                        print("Saving model checkpoint at EPOCH {}".format(episode_count))
                        checkpoint_name = "{}_epoch{:02d}".format(path_to_model,episode_count)
                        agent.save(checkpoint_name)
                    continue

                    # breaking here makes the code get out of the loop and never reaches the code under this statement.
                    # agent is always done in a single step so it is never possible to either replay or save

                if len(agent.memory) > args.batch_size:
                    agent.replay(args.batch_size)

        print("All episodes completed.")
        print("Saving final model")
        agent.save("{}_final".format(path_to_model))
        agent.printWeights()
    finally:
        output_file.close()
