import math


class Node:
    identifier = 0
    x = 0.0
    y = 0.0
    r = 50.0

    def __init__(self, identifier):
        self.identifier = identifier
        self.x, self.y = self.random_initial_location()
        print(str(self.x) + "," + str(self.y))

    def sees(self, neighbor):
        return math.sqrt(pow(self.x - neighbor.x, 2) + pow(self.y - neighbor.y, 2)) < self.r


    def random_initial_location(self):
        # TODO
        return (0,1)
    def move(self):
        # TODO
        pass

    def learn(self):
        # TODO
        pass

    def share(self, neighbor):
        # TODO
        pass

    def evaluate(self):
        # TODO
        pass
