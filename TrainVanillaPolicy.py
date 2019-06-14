import tensorflow as tf
from tqdm import tqdm
from src.Agent import Memory, VPGAgent
from src.BasicEnvironment import BasicEnvironment
import src.constants as c

if __name__ == "__main__":
    env = BasicEnvironment(ms_per_tick = c.TRAINING_DEFAULT_MS_PER_TICK, debug=False)
    tf.reset_default_graph() #Clear the Tensorflow graph.
    agent = VPGAgent(1e-2, env, 32, debug=False)

    jList, rList = agent.train(tqdm=tqdm, num_episodes=1000, checkpoint=None)
    with open("training_results.txt", "a") as f:
        f.write(" ".join(map(str, jList)))
        f.write("\n")
        f.write(" ".join(map(str, rList)))
 
