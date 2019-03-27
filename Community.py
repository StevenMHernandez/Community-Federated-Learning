import math
import random


class Community:
    x = 0.0
    y = 0.0
    r = 1000.0

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def getRandomLocationIn(self):
        a = random.random() * 2 * math.pi
        r = self.r * math.sqrt(random.random())
        x = r * math.cos(a)
        y = r * math.sin(a)
        return (x, y)