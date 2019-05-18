from tqdm import tqdm_notebook
import numpy as np
import tensorflow as tf
from functools import reduce # Valid in Python 2.6+, required in Python 3
import operator
from collections import deque
import random

import Utils
import tile_coding

class Discrete():
    def __init__(self, env):
        self.env = env
        #Initialize table with all zeros
        self.reset()
        # Set learning parameters
        self.lr = .8
        self.y = .95        
        
    def reset(self):
        self.Q = np.zeros([self.env.observation_space.n,self.env.action_space.n])

    def train(self, num_episodes = 2000):
        #create lists to contain total rewards and steps per episode
        jList = []
        rList = []
        for i in tqdm_notebook(range(num_episodes)):
            s = self.env.reset()
            rAll = 0
            d = False
            j = 0
            #The Q-Table learning algorithm
            while j < 99:
                j+=1
                #Choose an action by greedily (with noise) picking from Q table
                a = np.argmax(self.Q[s,:] + np.random.randn(1, self.env.action_space.n)*(1./(i+1)))
                #Get new state and reward from environment
                s1,r,d,_ = self.env.step(a)
                #Update Q-Table with new knowledge
                self.Q[s,a] = self.Q[s,a] + self.lr*(r + self.y*np.max(self.Q[s1,:]) - self.Q[s,a])
                rAll += r
                s = s1
                if d == True:
                    break
            jList.append(j)
            rList.append(rAll)
            
        return jList, rList
    
class DeepDiscrete():
    def __init__(self, env, y=.99, e=.1):
        self.env = env
        self.y = y
        self.e = e
        self.reset()
        
    class Model():
        def __init__(self, agent):
            #These lines establish the feed-forward part of the network used to choose actions
            self.tf_state = tf.placeholder(shape=[1,agent.env.nS],dtype=tf.float32)
            self.tf_linear_model = tf.layers.Dense(units=4, use_bias=True, kernel_initializer=tf.random_uniform_initializer(0, 0.01))
            self.tf_Q_predict = self.tf_linear_model(self.tf_state)
            self.tf_predict = tf.argmax(self.tf_Q_predict,1)

            #Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
            self.tf_target_q = tf.placeholder(shape=[1,agent.env.nA],dtype=tf.float32)
            self.loss = tf.reduce_sum(tf.square(self.tf_target_q - self.tf_Q_predict))
            self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
            self.updateModel = self.optimizer.minimize(self.loss)

            self.init = tf.global_variables_initializer()
        
    def reset(self):
        Utils.tf_reset()
        self.model = self.Model(self)
        
    def train(self, num_episodes=2000, max_steps=100, render=False):
        #create lists to contain total rewards and steps per episode
        num_steps_list = []
        reward_list = []
        with tf.Session() as sess:
            sess.run(self.model.init)
            for i in tqdm_notebook(range(num_episodes)):
                #Reset environment and get first new observation
                s = self.env.reset()
                total_reward = 0
                d = False
                j = 0
                #The Q-Network
                while j < max_steps:
                    j+=1
                    #Choose an action by greedily (with e chance of random action) from the Q-network
                    a,Q_pred = sess.run([self.model.tf_predict,self.model.tf_Q_predict],feed_dict={self.model.tf_state:np.identity(self.env.nS)[s:s+1]})
                    if np.random.rand(1) < self.e:
                        a[0] = self.env.action_space.sample()
                    #Get new state and reward from environment
                    s1,r,d,_ = self.env.step(a[0])

                    if render:
                        display.clear_output(wait=True)
                        print(f"Ep: {i}")
                        env.render()
                        time.sleep(.001)

                    #Obtain the Q' values by feeding the new state through our network
                    Q1 = sess.run(self.model.tf_Q_predict,feed_dict={self.model.tf_state:np.identity(16)[s1:s1+1]})
                    #Obtain maxQ' and set our target value for chosen action.
                    maxQ1 = np.max(Q1)
                    targetQ = Q_pred
                    targetQ[0,a[0]] = r + self.y*maxQ1
                    #Train our network using target and predicted Q values
                    _ = sess.run([self.model.updateModel],feed_dict={self.model.tf_state:np.identity(self.env.nS)[s:s+1],self.model.tf_target_q:targetQ})
                    total_reward += r
                    s = s1
                    if d == True:
                        #Reduce chance of random action as we train the model.
                        e = 1./((i/50) + 10)
                        break
                num_steps_list.append(j)
                reward_list.append(total_reward)
                if render:
                    time.sleep(.01)
        print("Percent of succesful episodes: " + str(sum(reward_list)/num_episodes * 100) + "%")
        return reward_list, num_steps_list

    
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
        return tf.one_hot(tile_coding.tiles(self.iht, self.num_tilings, state), self.iht.size, 0)
        
    def _calculate_state_shape(self):
        s = self.env.reset()
        return self._encode_state(s).shape
    
    def _build_model(self):
        Utils.tf_reset()
        
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(self.env.action_space.n, activation=tf.nn.relu))
        model.add(tf.keras.layers.Dense(self.env.action_space.n, activation=None))
        model.compile('sgd', loss=tf.keras.losses.MeanSquaredError())

        #tf_target_q = tf.placeholder(shape=[1,num_actions],dtype=tf.float32)
        return model
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((self._encode_state(state), action, reward, self._encode_state(next_state), done))
        
    def choose_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.env.action_space.n)
        encoded_state = self.encode_state(state)
        act_values = self.model.predict(encoded_state)
        return np.argmax(act_values[0])  # returns action
    
    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
              target = reward + self.gamma * \
                       np.amax(self.model.predict(next_state, steps=1)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
    def train(self, num_episodes, max_steps=500):
        num_steps_list = []
        reward_list = []
        with tf.Session() as sess:
            for i in tqdm_notebook(range(num_episodes)):
                #Reset environment and get first new observation
                s = self.env.reset()
                total_reward = 0
                d = False
                j = 0
                #The Q-Network
                for j in tqdm_notebook(range(max_steps)):
                    a = self.choose_action(s)
                    s1,r,d,_ = self.env.step(a)
                    self.remember(s, a, r, s1, d)
                    
                    s = s1
                    total_reward += r
                    if d == True:
                        #Reduce chance of random action as we train the model.
                        e = 1./((i/50) + 10)
                        break
                self.replay(32)
            print("Percent of succesful episodes: " + str(sum(reward_list)/num_episodes * 100) + "%")
            return reward_list, num_steps_list
        