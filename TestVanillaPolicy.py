import tensorflow as tf
from tqdm import tqdm
from src.Agent import Memory, VPGAgent
from src.BasicEnvironment import BasicEnvironment
import src.constants as c

if __name__ == "__main__":
    env = BasicEnvironment(ms_per_tick = c.MINECRAFT_DEFAULT_MS_PER_TICK, debug=False)
    tf.reset_default_graph() #Clear the Tensorflow graph.
    agent = VPGAgent(1e-2, env, 8, debug=False)

    #agent.train(tqdm=tqdm, num_episodes=500)
    reward_list, stepslist = agent.test(checkpoint=1000, tqdm=tqdm, num_episodes=20)
    with open("testing_results.txt", "a") as f:
        f.write(" ".join(map(str, reward_list)))
        f.write("\n")
        f.write(" ".join(map(str, stepslist)))
