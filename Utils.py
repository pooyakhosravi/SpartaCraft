import tensorflow as tf
import matplotlib.pyplot as plt

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
    ax[0].plot(j_list[0::5], label="Number of Steps")
    ax[0].set_title("Number of Steps")
    ax[1].plot(r_list[0::5], label="Rewards")
    ax[1].set_title("Total Reward")