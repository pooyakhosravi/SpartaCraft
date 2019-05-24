class ObservationSpace:
    def __init__(self, n):
        self.n = n

class BasicObservationSpace(ObservationSpace):
    def __init__(self, width, height):
        super(BasicObservationSpace, self).__init__(width * height)