import tensorflow as tf
from tqdm import tqdm
from src.Agent import Memory, PolicyAgent
from src.BasicEnvironment import BasicEnvironment
import src.constants as c

if __name__ == "__main__":
    env = BasicEnvironment(ms_per_tick = c.TRAINING_DEFAULT_MS_PER_TICK, debug=False)
    tf.reset_default_graph() #Clear the Tensorflow graph.
    agent = PolicyAgent(1e-2, env, 8, debug=False)

    agent.train(tqdm=tqdm, num_episodes=1000, checkpoint=300)
 