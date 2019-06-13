import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

def tf_reset():
    try:
        sess.close()
    except:
        pass
    tf.reset_default_graph()
    return tf.Session()

def graph_results(j_list, r_list):
    fig, ax = plt.subplots(2, 1, figsize=(24, 16))
    ax = ax.flat
    ax[0].plot(j_list, label="Number of Steps")
    ax[0].set_title("Number of Steps")
    ax[1].plot(r_list, label="Rewards")
    ax[1].set_title("Total Reward")

def discount(r, gamma):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size)):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r

def compute_gae(values, discounted_rewards, gamma, gae_lambda=1.00):
    gae = np.zeros_like(values)
    for i in reversed(range(0, discounted_rewards.size)):
        delta_t = discounted_rewards[i] + gamma * values[i + 1] - values[i]
        gae[i] = gae[i+1] * gamma * gae_lambda + delta_t
    return gae[:-1]
    