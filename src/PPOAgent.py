import tensorflow as tf

class PPOAgent():
    def __init__(self, lr, env, gamma=.99, epsilon=.1, debug=False):
        self.lr = lr
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
        self.debug = debug
        self._create_model()
        self.memory = Memory()

    def feed_dict(self, state_in, discounted_rewards, chosen_actions):
        return {

        }

    def _create_model(self):
        # Define Inputs
        s_size = self.env.observation_space.shape[0]
        a_size = self.env.action_space.n
        state_in_shape = [None,self.env.observation_space.shape[0]]
        self.state_in= tf.keras.Input(shape=[s_size],dtype=tf.float32)
        self.discounted_rewards_holder = tf.placeholder(shape=[None], dtype=tf.float32)
        self.action_probabilities =      tf.placeholder(shape=[None, a_size], dtype=tf.int32)
        self.old_neg_log_pac = tf.placeholder(tf.float32, [None])
        self.OLDVPRED = tf.placeholder(tf.float32, [None])

        # Feed Foward
        base = tf.keras.models.Sequential([
            tf.keras.layers.Dense(8, activation=tf.nn.relu, name='dense1'),
            tf.keras.layers.Dense(a_size, activation=tf.nn.softmax, name='dense2')
        ])
        actor = tf.keras.models.Sequential([
            tf.keras.layers.Dense(a_size, activation=tf.nn.softmax, name='actor')
        ])
        critic = tf.keras.models.Sequential([
            tf.keras.layers.Dense(1, activation=None, name='critic')
        ])
        self.base_output = base(self.state_in)
        self.actor_output  = actor(self.base)
        self.critic_output = critic(self.base)
        self.chosen_actions = tf.argmax(self.actor,1)

        # Loss
        advantage = self.discounted_rewards_holder - critic_output
        batch_size = tf.shape(self.actor)[0]
        num_outputs = tf.shape(self.actor)[1]
        chosen_action_holder = tf.argmax(self.action_probabilities, axis=1)
        indexes = tf.range(0, batch_size) * num_outputs + chosen_action_holder
        responsible_outputs = tf.gather(tf.reshape(self.actor, [-1]), indexes)

        # Calculate ratio (pi current policy / pi old policy)
        ratio = tf.exp(self.OLDNEGLOGPAC - neglogpac)
        self.actor_loss = -tf.reduce_mean(clipped_obj)
        self.critic_loss = -tf.reduce_mean(tf.log(self.responsible_outputs) * self.advantage)
        self.entropy = tf.reduce_mean(dist.entropy(actor))
        self.clipped_frac = _clipped_frac(new_log_probs, old_log_probs, self._advs, self._epsilon)
        self.objective = (entropy_reg * self.entropy - self.actor_loss -
                          vf_coeff * self.critic_loss)

        # Defining Loss = - J is equivalent to max J
        pg_losses = -ADV * ratio
        pg_losses2 = -ADV * tf.clip_by_value(ratio, 1.0 - CLIPRANGE, 1.0 + CLIPRANGE)
        # Final PG loss
        pg_loss = tf.reduce_mean(tf.maximum(pg_losses, pg_losses2))
        approxkl = .5 * tf.reduce_mean(tf.square(neglogpac - OLDNEGLOGPAC))
        clipfrac = tf.reduce_mean(tf.to_float(tf.greater(tf.abs(ratio - 1.0), CLIPRANGE)))
        
        
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
