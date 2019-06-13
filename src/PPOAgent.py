import tensorflow as tf
from src.Rollout import Rollout
import numpy as np
from src.Utils import *

class PPOAgent():
    def __init__(self, lr, env, gamma=.99, epsilon=.1, entropy_coeff=.01, cliprange=.2, debug=False):
        self.lr = lr
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
        self.entropy_coeff = entropy_coeff
        self.cliprange = cliprange
        self.debug = debug
        self._create_model()
        self.rollout = Rollout()

    def feed_dict(self, state_in, discounted_rewards, gae, action_probabilities):
        ret = {}
        ret[self.state_in] = state_in
        ret[self.discounted_rewards] = discounted_rewards
        ret[self.gae] = gae
        ret[self.action_probabilities] = action_probabilities
        return ret

    def _create_model(self):
        # Define Inputs
        s_size = self.env.observation_space.shape[0]
        a_size = self.env.action_space.n
        self.state_in             = tf.keras.Input(shape=[s_size],dtype=tf.float32, name="state_in")
        self.discounted_rewards   = tf.placeholder(shape=[None], dtype=tf.float32, name="discounted_rewards")
        self.gae                  = tf.placeholder(shape=[None], dtype=tf.float32, name="gae")
        self.action_probabilities = tf.placeholder(shape=[None, a_size], dtype=tf.float32, name="action_probabilities")
        # Keep track of old actor
        self.OLDNEGLOGPAC = tf.placeholder(tf.float32, [None])
        # Keep track of old critic
        self.OLDVPRED = tf.placeholder(tf.float32, [None])
        self.CLIPRANGE = tf.constant(self.cliprange)

        # Feed Foward
        base = tf.keras.models.Sequential([
            tf.keras.layers.Dense(8, activation=tf.nn.relu, name='base_dense1'),
            tf.keras.layers.Dense(8, activation=tf.nn.relu, name='base_dense2'),
        ])
        actor = tf.keras.layers.Dense(a_size, activation=tf.nn.softmax, name='actor_output')
        critic = tf.keras.layers.Dense(1, activation=None, name='critic_output')
        self.base_output = base(self.state_in)
        self.actor_output = actor(self.base_output)
        self.critic_output = critic(self.base_output)

        # Loss
        batch_size = tf.shape(self.actor_output)[0]
        num_outputs = tf.shape(self.actor_output)[1]
        chosen_action_holder = tf.argmax(self.action_probabilities, axis=1, output_type=tf.int32)
        indexes = tf.range(0, batch_size) * num_outputs + chosen_action_holder
        responsible_outputs = tf.gather(tf.reshape(self.actor_output, [-1]), indexes)

        # Calculate ratio (pi current policy / pi old policy
        self.actor_loss = -tf.reduce_sum(tf.log(responsible_outputs) * self.gae)
        self.critic_loss = 0.5 * tf.reduce_sum(tf.square(self.discounted_rewards - self.critic_output))
        self.entropy = -tf.reduce_sum(self.actor_output * tf.log(self.actor_output))
        self.loss = self.actor_loss + 0.5 * self.critic_loss - self.entropy_coeff * self.entropy
        
        # Update (Computed Per n episodes)
        # Placeholders for trainable variable gradients every n episodes
        trainable_variables = tf.trainable_variables()
        self.gradient_holders = []
        for idx, var in enumerate(trainable_variables):
            placeholder = tf.placeholder(tf.float32,name=str(idx)+'_holder')
            self.gradient_holders.append(placeholder)
        
        # Compute gradients based on single episode loss
        self.gradients = tf.gradients(self.loss, trainable_variables)
        
        # Update gradients assuming gradient_holders are filled with gradBuffer
        optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        self.update_batch = optimizer.apply_gradients(zip(self.gradient_holders,trainable_variables))

    def choose_action(self, sess, s):
        #Probabilistically pick an action given our network outputs.
        a_dist, value = sess.run([self.actor_output, self.critic_output], feed_dict={self.state_in:[s]})
        # if self.debug:
        #     print(f"s: {s} | a_dist: {a_dist}")
        a = np.random.choice(a_dist[0],p=a_dist[0])
        a = np.argmax(a_dist == a)
        return a, a_dist, value

    def update(self, sess, gradBuffer, i, update_frequency=10):
        rewards = np.array(self.rollout['reward']).squeeze()
        values = np.array(self.rollout['value'] + [0]).squeeze()
        states = np.vstack(np.array(self.rollout['state']).squeeze())
        a_dists = np.array(self.rollout['a_dist']).squeeze()

        discounted_rewards = discount(rewards, self.gamma)
        gae = compute_gae(values, discounted_rewards, self.gamma)
        feed_dict = self.feed_dict(states, discounted_rewards, gae, a_dists)
        grads, actor_loss, critic_loss, loss  = sess.run([self.gradients, self.actor_loss, self.critic_loss, self.loss], feed_dict=feed_dict)
        for idx,grad in enumerate(grads):
            gradBuffer[idx] += grad
            grad_nan_count = np.count_nonzero(np.isnan(grad))
            if grad_nan_count > 0:
                print(f'grad nan count: {grad_nan_count}')

        if i % update_frequency == 0 and i != 0:
            feed_dict = dict(zip(self.gradient_holders, gradBuffer))
            _ = sess.run(self.update_batch, feed_dict=feed_dict)
            for ix,grad in enumerate(gradBuffer):
                gradBuffer[ix] = grad * 0
        return actor_loss, critic_loss, loss

    def train(self, tqdm, checkpoint = None, num_episodes=5000, max_ep=999, print_step=50):
        saver = tf.train.Saver()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth=True
        total_reward = []
        total_length = []
        total_actor_loss = []
        total_critic_loss = []
        total_loss = []
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
                self.rollout.clear()
                running_reward = 0
                for j in range(max_ep):
                    a, a_dist, value = self.choose_action(sess, s)
                    s1,r,d,_ = self.env.step(a)
                    self.rollout.remember(s, a_dist, value, r, s1, d)
                    s = s1
                    running_reward += r
                    if d:
                        # Update the Network and our running tally
                        actor_loss, critic_loss, loss = self.update(sess, gradBuffer, i)
                        total_actor_loss.append(actor_loss)
                        total_critic_loss.append(critic_loss)
                        total_loss.append(loss)
                        total_reward.append(running_reward)
                        total_length.append(j)
                        if i % print_step == 0 and i != checkpoint:
                            print(f'Avg reward over last {print_step} eps: {np.mean(total_reward[-print_step:])}')
                            save_path = saver.save(sess, f"./checkpoints/model{i}.ckpt")
                            print(f"Model saved in path: {save_path} at episode: {i}")
                        break
        return total_length, total_reward, total_actor_loss, total_critic_loss, total_loss
