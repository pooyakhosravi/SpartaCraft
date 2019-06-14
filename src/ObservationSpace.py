class ObservationSpace:
    def __init__(self, n, shape=()):
        self.n = n
        self.shape=shape

class BasicObservationSpace(ObservationSpace):
    def __init__(self, n, shape):
        super(BasicObservationSpace, self).__init__(n, shape)