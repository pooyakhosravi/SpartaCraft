from collections import deque
import random
import numpy as np

class Rollout():
    def __init__(self, max_len=2000):
        self.clear()
        
    def clear(self):
        self.history = {
            "state": [],
            "a_dist": [],
            "value": [],
            "reward": [],
            "next_state": [],
            "done": []
        }

    def remember(self, state, a_dist, value, reward, next_state, done):
        self.history['state'].append(state)
        self.history['a_dist'].append(a_dist)
        self.history['value'].append(value)
        self.history['reward'].append(reward)
        self.history['next_state'].append(next_state)
        self.history['done'].append(done)

    def get_minibatch(self, batch_size):
        return random.sample(self.history, min(batch_size, len(self.history)))

    def __getitem__(self, key):
        return self.history[key]
