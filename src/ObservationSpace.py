class ObservationSpace:
    def __init__(self, n, shape=()):
        self.n = n
        self.shape=shape

class BasicObservationSpace(ObservationSpace):
    def __init__(self, width, height):
        super(BasicObservationSpace, self).__init__(None, (2,))