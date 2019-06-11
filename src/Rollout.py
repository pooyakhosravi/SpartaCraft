class Rollout():
    def __init__(self, max_len=2000):
        self.history = deque(maxlen=max_len)

    def clear(self):
        self.history.clear()

    def remember(self, state, actions, reward, next_state, done):
        self.history.append((state, actions, reward, next_state, done))

    def get_minibatch(self, batch_size):
        return random.sample(self.history, min(batch_size, len(self.history)))

def discount_rewards(r, gamma):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size)):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r