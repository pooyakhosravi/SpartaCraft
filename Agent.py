from tqdm import tqdm_notebook
import numpy as np
import tensorflow as tf
from functools import reduce # Valid in Python 2.6+, required in Python 3
import operator
from collections import deque
import random
import time
from IPython.display import clear_output

import Utils
import tile_coding

class QNetwork():
    def __init__(self, env):
        self.model = self._build_model(env)
        
    def _build_model(self, env):
        Utils.tf_reset()
        
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(env.action_space.n, activation=None, use_bias=False))
        model.compile('sgd', loss=tf.keras.losses.MeanSquaredError())

        return model
    
    def _encode_state(self, s):
        encoded_state = np.expand_dims(np.array([s]), 0)
        print(encoded_state, type(encoded_state), encoded_state.shape)
        return encoded_state

    def predict(self, s):
        return self.model.predict(self._encode_state(s), batch_size=1, steps=1)[0]
    
    def update(self, s, a, s1, r, lr, y):
        target_f = self.model.predict(self._encode_state(s))
        target = (1.0-lr)*target_f[0][a] + lr*(r+y*np.max(self.predict(s1)))
        target_f[0][a] = target
        self.model.fit(self._encode_state(s), target_f, epochs=1, verbose=0, steps_per_epoch=1)

class QTable():
    def __init__(self, env):
        self.reset(env)
        
    def reset(self, env):
         #Initialize table with all zeros
        self.Q = np.zeros(shape=(env.observation_space.n, env.action_space.n))
        
    def predict(self, s):
        return self.Q[s,:]
    
    def update(self, s, a, s1, r, lr, y):
        self.Q[s,a] = (1.0-lr)*self.Q[s,a] + lr*(r+y*np.max(self.predict(s1)))
        
    def render(self):
        print(self.Q)

class QLearning():
    def __init__(self, env, q_function):
        self.env = env
        # Set learning parameters
        self.lr = .8
        self.y = .95
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.q_function = q_function
        
    def reset(self):
        self.Q = np.zeros(shape=(self.env.observation_space.n, self.env.action_space.n))
   
    def choose_action(self, s, epsilon=None):
        # Assuming we have a decaying epsilon, we start by picking actions randomly and then pick based on q function
        epsilon = epsilon if epsilon else self.epsilon
        if np.random.rand() < epsilon:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.q_function.predict(s))

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
                a = self.choose_action(s)
                s1,r,d,_ = self.env.step(a)
                
                # Update Q-Table with new knowledge and prepare for next iteration
                target = self.q_function.predict(s1)
                self.q_function.update(s, a, s1, r, self.lr, self.y)
                rAll += r
                s = s1
                if d:
                    break
                    
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

            jList.append(j)
            rList.append(rAll)
            
        return jList, rList
    
    def test(self, num_episodes=1):
        reward_list = []
        for ep in tqdm_notebook(range(num_episodes)):
            d = False
            s = self.env.reset()
            while not d:
                clear_output(wait=True)
                self.env.render()
                time.sleep(.005)
                a = self.choose_action(s, 0.0)
                s,r,d,_ = self.env.step(a)
            clear_output(wait=True)
            self.env.render()
            time.sleep(.25)
            reward_list.append(r)
        return reward_list

def MountainCarScaling(env, s):
    return 10.0 * s / [env.max_position - env.min_position, env.max_speed + env.max_speed]
    
class DeepContinuous():
    def __init__(self, env, tile_coding_size = 2048):
        self.env = env
        self.iht = tile_coding.IHT(tile_coding_size)
        self.num_tilings = 4 * reduce(operator.mul, env.observation_space.shape, 1)
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()
        self.state_shape = self._calculate_state_shape()
        
    def _encode_state(self, state):
        return state
        scaled_state = MountainCarScaling(self.env, state)
        return tf.one_hot(tile_coding.tiles(self.iht, self.num_tilings, scaled_state), self.iht.size, dtype=tf.float32)
        
    def _calculate_state_shape(self):
        s = self.env.reset()
        return self._encode_state(s).shape
    
    def _build_model(self):
        Utils.tf_reset()
        
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(self.env.action_space.n, activation=None, use_bias=False))
        model.compile('sgd', loss=tf.keras.losses.MeanSquaredError())

        return model
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((self._encode_state(state), action, reward, self._encode_state(next_state), done))
        
    def choose_action(self, state, epsilon=None):
        epsilon = epsilon if epsilon else self.epsilon
        if np.random.rand() <= epsilon:
            return random.randrange(self.env.action_space.n)
        encoded_state = self._encode_state(state)
        act_values = self.model.predict(encoded_state, batch_size=1, steps=1)
        return np.argmax(act_values[0])  # returns action
    
    def replay(self, batch_size):
        minibatch = random.sample(self.memory, min(batch_size, len(self.memory)))
        
        for state, action, reward, next_state, done in minibatch:
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state, batch_size=1, steps=1)[0])
            else:
                target = reward
            target_f = self.model.predict(state, batch_size=1, steps=1)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0, steps_per_epoch=1)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
    def train(self, num_episodes, max_steps=500):
        num_steps_list = []
        reward_list = []
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for i in tqdm_notebook(range(num_episodes)):
                #Reset environment and get first new observation
                s = self.env.reset()
                total_reward = 0
                d = False
                j = 0
                #The Q-Network
                for j in range(max_steps):
                    a = self.choose_action(s)
                    s1,r,d,_ = self.env.step(a)
                    self.remember(s, a, r, s1, d)
                    
                    s = s1
                    total_reward += r
                    if d == True:
                        break
                reward_list.append(total_reward)
                num_steps_list.append(j)
                self.replay(32)
            print("Percent of succesful episodes: " + str(sum(reward_list)/num_episodes * 100) + "%")
            return num_steps_list, reward_list
        
    def test(self):
        with tf.Session() as sess:
            s = self.env.reset()
            d = False
            while not d:
                a = self.choose_action(s, 0.0)
                s1,r,d,_ = self.env.step(a)
                s = s1
                
                self.env.render()
                time.sleep(.25)
            
            