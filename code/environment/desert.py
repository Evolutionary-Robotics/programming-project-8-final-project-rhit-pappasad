import numpy as np

DEF_DESERT_COLOR = (245, 202, 86)

class Desert:

    def __init__(self, size):
        self.size = size
        self.min_x = 0
        self.min_y = 0
        self.max_x = size[0]
        self.max_y = size[1]
        self.camels = []
        self.pools = []
        self.worms = []
        self.max_distance = np.sqrt(self.size[0]**2 + self.size[1]**2)

    def addCamel(self, camel):
        self.camels.append(camel)

    def addWorm(self, worm):
        self.worms.append(worm)

    def addPool(self, pool):
        self.pools.append(pool)

    def updateAgents(self, stepsize):
        cam_, worm_ = 0.0, 0.0
        for camel in self.camels:
            cam_ = self.updateCamel(camel)
        for worm in self.worms:
            worm_ = worm.handleUpdate(stepsize, self.camels, (self.min_x, self.min_y), (self.max_x, self.max_y))
        for pool in self.pools:
            pass #expel

        return cam_, worm_

    def updateCamel(self, camel):
        pass



