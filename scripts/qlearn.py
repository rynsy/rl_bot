import random
import numpy as np
import time, pickle, os
import sys

class QLearn:
    def __init__(self, states, actions, epsilon, alpha, gamma):
        self.q = np.zeros((states, actions))
        self.epsilon = epsilon  
        self.alpha = alpha     
        self.gamma = gamma    
        self.actions = actions

    def chooseAction(self, state, return_q=False):
        action = 0
        if np.random.uniform(0, 1) < self.epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(self.q[state, :])
        if return_q:
            return action, self.q[state][action]
        else:
            return action

    def learn(self, state, action, reward, state2):
        predict = self.q[state,action]
        target = reward + self.gamma * np.max(self.q[state2, :])
        self.q[state, action] = self.q[state, action] + self.alpha * (target - predict)
