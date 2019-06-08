import random
import gym
import numpy as np
from collections import deque
from scipy.special import softmax
import random
from sklearn import preprocessing
import matplotlib.pyplot as plt
from keras.models import Sequential
import time

import os

# Deep Q-learning Agent
class DQNAgent:
    def __init__(self, state_size, action_size, memsize = 3000, ga = 0.95, explore_rate = 1, explore_decay = 0.9995):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen = memsize)
        self.gamma = ga    # discount rate
        self.epsilon = explore_rate  # exploration rate
        self.epsilon_min = 0.3
        self.epsilon_decay = explore_decay
        self.target_model = Sequential()
        self.engine_model = Sequential()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            chose = np.random.randint(0,4)
            return chose
        
        act_values = 0
        act_values = self.model.predict(state)
        chose = np.argmax(act_values)
        if chose == 0:
            return chose
        if chose == 1:
            return chose
        if chose == 2:
            return chose
        if chose == 3:
            return chose
        return act_values
    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        xs = []
        ys = []

        xs = minibatch[:][0]
        ys = reward/10 +np.multiply(self.gamma, self.model.predict(minibatch[:][3]))
        xs = np.array(xs)
        ys = np.array(ys)
        self.model.fit(xs, ys, epochs= 1, verbose=0 , batch_size=batch_size)
                
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def save_model(self, model_name = './checkpoint.h5', mem_name = 'memory'):
        self.model.save(model_name)
        np.save(mem_name , self.memory)

    def load_model(self,  model_name = './checkpoint.h5' , mem_name = 'memory.npy'):
        self.model.load_weights(model_name)
        self.memory = np.load(mem_name, allow_pickle=True)
        self.memory = deque(self.memory)
    def learn (self):
        xs = self.memory[:][0]
        ys = self.memory[:][3]
        self.engine_model.predict(
        xs = np.array(xs)
        ys = np.array(ys)