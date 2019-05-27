import tensorflow as tf
from tqdm import tqdm
from src.Agent import Memory, PolicyAgent
from src.BasicEnvironment import BasicEnvironment

if __name__ == "__main__":
    env = BasicEnvironment(debug=False)
    tf.reset_default_graph() #Clear the Tensorflow graph.
    agent = PolicyAgent(1e-2, env, 8, debug=False)

    agent.train(tqdm)
