import random

class ActionSpace:
    def __init__(self, actions):
        self.actions = actions
        self.size = len(actions)
    
    def random(self):
        return random.choice(self.actions)

    
