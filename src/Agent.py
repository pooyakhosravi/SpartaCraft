from tqdm import tqdm_notebook
import numpy as np
import tensorflow as tf
from functools import reduce # Valid in Python 2.6+, required in Python 3
import operator
from collections import deque
import random
import time
from IPython.display import clear_output

from src import Utils
from src import tile_coding

class QNetwork():
    def __init__(self, env):
        self.env = env
        self.model = self._build_model()
        
    def _build_model(self):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(self.env.action_space.n, activation=None, use_bias=False))
        model.compile('SGD', loss=tf.keras.losses.MeanSquaredError())
        return model
    
    def _encode_state(self, s):
        encoded_state = np.identity(self.env.observation_space.n)[s:s+1]
        #encoded_state = tf.one_hot([s], self.env.observation_space.n)
        #print(f"encoded_state shape: {encoded_state.shape} \n encoded_state: {encoded_state}")
        return encoded_state

    def predict(self, s):
        prediction = self.model.predict(s, batch_size=1, steps=1)[0]
        #print(f"prediction: {prediction}, shape: {prediction.shape}")
        return prediction
    
    def update(self, s, a, target, lr):
        target_f = self.model.predict(s, batch_size=1, steps=1)
        #print(f"target_f shape: {target_f.shape}, target_f[0] shape: {target_f[0].shape}")
        target_f[0][a] = target
        self.model.fit(s, target_f, epochs=1, verbose=0, steps_per_epoch=1)
        
    def render(self):
        for s in range(self.env.observation_space.n):
            print(self.predict(self._encode_state(s)))

class QTable():
    def __init__(self, env):
        self.reset(env)
        
    def reset(self, env):
         #Initialize table with all zeros
        self.Q = np.zeros(shape=(env.observation_space.n, env.action_space.n))
        
    def predict(self, s):
        return self.Q[s,:]
    
    def update(self, s, a, target, lr):
        self.Q[s,a] = (1.0-lr)*self.Q[s,a] + lr * target
        
    def render(self):
        print(self.Q)
        
    def _encode_state(self, s):
        return s

class QLearning():
    def __init__(self, env, q_function):
        self.env = env
        # Set learning parameters
        self.lr = .2
        self.y = .99
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.q_function = q_function
        self.memory = deque(maxlen=2000)
        
    def reset(self):
        self.Q = np.zeros(shape=(self.env.observation_space.n, self.env.action_space.n))
   
    def choose_action(self, s, training=True):
        # Assuming we have a decaying epsilon, we start by picking actions randomly and then pick based on q function
        if training and np.random.rand() < self.epsilon:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.q_function.predict(self.q_function._encode_state(s)))
        
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((self.q_function._encode_state(state), action, reward, self.q_function._encode_state(next_state), done))
        
    def replay(self, batch_size):
        minibatch = random.sample(self.memory, min(batch_size, len(self.memory)))
        for state, action, reward, next_state, done in minibatch:
            if not done:
                target = reward + self.y * np.amax(self.q_function.predict(next_state))
            else:
                target = reward
            self.q_function.update(state, action, target, self.lr)
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def train(self, num_episodes = 2000, max_steps = 100):
        # create lists to contain total rewards and steps per episode
        jList = []
        rList = []
        for ep in tqdm_notebook(range(num_episodes)):
            s = self.env.reset()
            rAll = 0
            d = False
            for j in range(1, max_steps):
                # Take action and get new state and reward from environment
                a = self.choose_action(s, training=True)
                s1,r,d,_ = self.env.step(a)
                
                # 'Member
                self.remember(s, a, r, s1, d)
                rAll += r
                s = s1
                if d:
                    break
            #Save results from episode and train on memory
            jList.append(j)
            rList.append(rAll)
            self.replay(32)
            
        return jList, rList
    
    def test(self, num_episodes=1, render=False):
        reward_list = []
        for ep in tqdm_notebook(range(num_episodes)):
            d = False
            s = self.env.reset()
            while not d:
                if render:
                    clear_output(wait=True)
                    self.env.render()
                    time.sleep(.005)
                a = self.choose_action(s, training=False)
                s,r,d,_ = self.env.step(a)
            if render:
                clear_output(wait=True)
                self.env.render()
                time.sleep(.25)
            reward_list.append(r)
        return reward_list
    
    def render(self):
        self.q_function.render()

def MountainCarScaling(env, s):
    return 10.0 * s / [env.max_position - env.min_position, env.max_speed + env.max_speed]


class Random():
    def __init__(self, env):
        self.env = env

    def choose_action(self, s):
        return self.env.action_space.sample()
        
    def test(self, num_episodes=1, render=False):
        reward_list = []
        for ep in tqdm_notebook(range(num_episodes)):
            d = False
            s = self.env.reset()
            while not d:
                a = self.choose_action(s)
                s,r,d,_ = self.env.step(a)
            print(f"Final: state: {s}, reward: {r}, done: {d}")
            reward_list.append(r)
        return reward_list