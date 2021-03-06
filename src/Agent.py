from tqdm import tqdm, tqdm_notebook
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
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


class Memory():
    def __init__(self, max_len=2000):
        self.max_len = max_len
        self.history = {}
        self.idx = 0
        for key in ['action', 'reward']:
            self.history[key] = np.zeros(max_len)
        self.history['grid_world'] = np.zeros((max_len, 20,20))
        self.history['player_info'] = np.zeros((max_len, 4))

    def clear(self):
        self.idx = 0

    def remember(self, state, action, reward):
        if self.idx >= self.max_len:
            pass
        self.history['grid_world'][self.idx] = state[0]
        self.history['player_info'][self.idx] = state[1]
        self.history['action'][self.idx] = action
        self.history['reward'][self.idx] = reward
        self.idx += 1

    def __getitem__(self, key):
        return self.history[key][:self.idx]

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
        self.memory = Memory()
        
    def reset(self):
        self.Q = np.zeros(shape=(self.env.observation_space.n, self.env.action_space.n))
   
    def choose_action(self, s, training=True):
        # Assuming we have a decaying epsilon, we start by picking actions randomly and then pick based on q function
        if training and np.random.rand() < self.epsilon:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.q_function.predict(self.q_function._encode_state(s)))
        
    def replay(self, batch_size):
        minibatch = self.memory.get_minibatch(batch_size)
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
                self.memory.remember(s, a, r, s1, d)
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

def discount_rewards(r, gamma):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size)):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r

class Random():
    def __init__(self, env):
        self.env = env

    def choose_action(self, s):
        return self.env.action_space.sample()
        
    def test(self, num_episodes=1, render=False):
        reward_list = []
        for ep in tqdm_notebook(range(num_episodes)):
            d = False
            reward = 0
            s = self.env.reset()
            while not d:
                a = self.choose_action(s)
                s,r,d,_ = self.env.step(a)
                reward += r
                print(f'{r}, ', end='')
            print(f"Final: state: {s}, reward: {r}, done: {d}")
            reward_list.append(reward)
        return reward_list

class PolicyAgent():
    def __init__(self, env, lr, gamma, debug=False):
        self.lr = lr
        self.env = env
        self.a_size = self.env.action_space.n
        self.gamma = gamma
        self.debug = debug
        self.memory = Memory()
        self.output = None # Child must implement

    def choose_action(self, sess, s):
        #Probabilistically pick an action given our network outputs.
        grid_world, player_info = s
        a_dist = sess.run(self.output,feed_dict={self.grid_world:[grid_world],self.player_info:[player_info]})
        if self.debug:
            print(f"a_dist: {a_dist}")
        a = np.random.choice(a_dist[0],p=a_dist[0])
        a = np.argmax(a_dist == a)
        return a

    def test(self, checkpoint, tqdm=tqdm, num_episodes=1, max_ep=999):
        saver = tf.train.Saver()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth=True
        total_reward = []
        total_length = []
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            saver.restore(sess, f"./checkpoints/model{checkpoint}.ckpt")
            for i in tqdm(range(num_episodes)):
                s = self.env.reset()
                running_reward = 0
                for j in range(max_ep):
                    a = self.choose_action(sess, s)
                    s1,r,d,_ = self.env.step(a)
                    s = s1
                    running_reward += r
                    if d:
                        # Update the Network and our running tally
                        total_reward.append(running_reward)
                        total_length.append(j)
                        break

        return total_reward, total_length

class VPGAgent(PolicyAgent):
    def __init__(self, lr, env, hidden_layer_size, gamma=.99, debug=False):
        PolicyAgent.__init__(self, env, lr, gamma, debug)
        self.h_size = hidden_layer_size
        self._create_model()
        
    def _create_model(self):
        self.grid_world = tf.keras.Input(shape=self.env.observation_space.shape, dtype=tf.float32, name='grid_world')
        self.player_info = tf.keras.Input(shape=[self.env.observation_space.n], dtype=tf.float32, name='player_info')        
        # Feed Foward
        grid_world_feature_extraction = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(64, 3, activation=tf.nn.relu, name='cnn1'),
            tf.keras.layers.MaxPool2D(),
            tf.keras.layers.Dropout(.2),
            tf.keras.layers.Conv2D(32, 3, activation=tf.nn.relu, name='cnn2'),
            tf.keras.layers.MaxPool2D(),
            tf.keras.layers.Dropout(.2),
            tf.keras.layers.Flatten(),
        ])
        model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(self.h_size, activation=tf.nn.relu, name='dense1'),
            tf.keras.layers.Dense(self.a_size, activation=tf.nn.softmax, name='dense2')
        ])
        grid_world_features = grid_world_feature_extraction(tf.expand_dims(self.grid_world, -1))
        print(f"grid_world_features.shape: {grid_world_features.shape}, player_info.shape: {self.player_info.shape}")
        feature_vec = tf.concat(axis=1, values=[grid_world_features, self.player_info],)
        self.output = model(feature_vec)
        print(f"Output Shape: {self.output.shape}")
        self.chosen_action = tf.argmax(self.output,1)

        # Loss (Computed Per episode)
        self.discounted_rewards_holder = tf.placeholder(shape=[None], dtype=tf.float32)
        self.chosen_action_holder =      tf.placeholder(shape=[None], dtype=tf.int32)
        
        batch_size = tf.shape(self.output)[0]
        num_outputs = tf.shape(self.output)[1]
        self.indexes = tf.range(0, batch_size) * num_outputs + self.chosen_action_holder
        self.responsible_outputs = tf.gather(tf.reshape(self.output, [-1]), self.indexes)

        self.advantage = self.discounted_rewards_holder

        self.loss = -tf.reduce_mean(tf.log(self.responsible_outputs) * self.advantage)
        
        # Update (Computed Per n episodes)
        # Placeholders for trainable variable gradients every n episodes
        trainable_variables = tf.trainable_variables()
        self.gradient_holders = []
        for idx,var in enumerate(trainable_variables):
            placeholder = tf.placeholder(tf.float32,name=str(idx)+'_holder')
            self.gradient_holders.append(placeholder)
        
        # Compute gradients based on single episode loss
        self.gradients = tf.gradients(self.loss,trainable_variables)
        
        # Update gradients assuming gradient_holders are filled with gradBuffer
        optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        self.update_batch = optimizer.apply_gradients(zip(self.gradient_holders,trainable_variables))
    
    def update(self, sess, gradBuffer, i, update_frequency=5):
        rewards = self.memory['reward']
        actions = self.memory['action']
        grid_world = self.memory['grid_world']
        player_info = self.memory['player_info']
        discounted_rewards = discount_rewards(rewards, self.gamma)
        feed_dict={self.discounted_rewards_holder:discounted_rewards,
                self.chosen_action_holder:actions,
                self.grid_world:grid_world,
                self.player_info:np.vstack(player_info)}
        grads = sess.run(self.gradients, feed_dict=feed_dict)
        for idx,grad in enumerate(grads):
            gradBuffer[idx] += grad

        if i % update_frequency == 0 and i != 0:
            feed_dict = dict(zip(self.gradient_holders, gradBuffer))
            _ = sess.run(self.update_batch, feed_dict=feed_dict)
            for ix,grad in enumerate(gradBuffer):
                gradBuffer[ix] = grad * 0

    def train(self, checkpoint = None, tqdm=tqdm, num_episodes=5000, max_ep=999):
        saver = tf.train.Saver()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth=True
        total_reward = []
        total_length = []
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            if checkpoint:
                saver.restore(sess, f"./checkpoints/model{checkpoint}.ckpt")
            else:
                checkpoint = 1
            gradBuffer = sess.run(tf.trainable_variables())
            for ix,grad in enumerate(gradBuffer):
                gradBuffer[ix] = grad * 0
                
            for i in tqdm(range(checkpoint, num_episodes + 1)):
                s = self.env.reset()
                self.memory.clear()
                running_reward = 0
                for j in range(max_ep):
                    a = self.choose_action(sess, s)
                    s1,r,d,_ = self.env.step(a)
                    self.memory.remember(s, a, r)
                    s = s1
                    running_reward += r
                    if d:
                        # Update the Network and our running tally
                        self.update(sess, gradBuffer, i)
                        total_reward.append(running_reward)
                        total_length.append(j)
                        if i % 50 == 0 and i != checkpoint:
                            print(np.mean(total_reward[-50:]))
                            save_path = saver.save(sess, f"./checkpoints/model{i}.ckpt")
                            print(f"Model saved in path: {save_path} at episode: {i}")
                        break
        return total_length, total_reward