import random

class ActionSpace:
    def __init__(self, actions):
        self.actions = actions
        self.n = len(actions)
    
    def sample(self):
        return random.randint(0, self.n - 1)
    
class BasicActionSpace(ActionSpace):
    def __init__(self):
        moves = ["forward 1", "back 1", "left 1", "right 1", "attack 1", "attack 0"]
        super(BasicActionSpace, self).__init__(moves)
        