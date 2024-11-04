import numpy as np
import bodies

DESERT_SIZE = (800, 800)
DESERT_COLOR = (245, 202, 86)

class Desert:
    def_size = DESERT_SIZE
    color = DESERT_COLOR
    _agents = ('camel', 'worm')
    def_pool_size = (50.0, 50.0)

    def __init__(self, size=None):
        self.min_x = 0
        self.min_y = 0
        if size is None:
            size = self.def_size
        self.size = size
        self.max_x = self.size[0]
        self.max_y = self.size[1]
        self.camel = None
        self.worm = None
        self.pool = None
        self.max_distance = np.sqrt(self.size[0]**2 + self.size[1]**2)

    def addAgent(self, agent):
        if isinstance(agent, bodies.Sandworm):
            self.worm = agent
        else:
            self.camel = agent

    def addPool(self, x, y, size=def_pool_size):
        self.pool = Pool(x, y, size)

    def updateAgents(self, stepsize, action):
        ret = 0.0
        if self.camel is not None:
            ret = self.camel.handleUpdate(stepsize, action)
            if self.pool is not None:
                self.pool.expelAgent(self.camel)
            if self.worm is not None:
                self.worm.handleUpdate(stepsize, self.camel)
                if self.pool is not None:
                    self.pool.expelAgent(self.worm)
        return ret

class Pool:
    color = (0, 0, 255)

    def __init__(self, x, y, size):
        self.x = x
        self.y = y
        self.size = size
        self.radius = np.sqrt(size[0]**2 + size[1]**2)

    def getShape(self):
        return 'Circle', self.color, (self.x, self.y), self.radius

    def expelAgent(self, agent: bodies.Body):
        distance = agent.getDistance(self.x, self.y)
        if distance < self.radius:
            expel_direction = -agent.getDirectionTo(self.x, self.y) #rel to pool center
            agent.x += np.cos(expel_direction)*(self.radius - distance)
            agent.y += np.sin(expel_direction)*(self.radius - distance)
            if hasattr(agent, 'reached_water'):
                agent.reached_water = True
            agent.angle = expel_direction





if __name__ == '__main__':
    import matplotlib.pyplot as plt
    test = Desert()
    camel = bodies.Camel(0, -350, 0, test)
    worm = bodies.Sandworm(0, 350, 0, test)
    length = 10000
    distances = []
    states = []
    CamXs = np.zeros(length)
    CamYs = np.zeros(length)
    WormXs = np.zeros(length)
    WormYs = np.zeros(length)
    stepsize = 1
    i = 0
    while i < length and camel.is_alive:
        if i % 100 == 0:
            pass
        #distances.append(camel.getDistance(worm.x, worm.y))
        #states.append(250 * int(worm.state == 'Pursuit'))
        CamXs[i] = camel.x
        CamYs[i] = camel.y
        WormXs[i] = worm.x
        WormYs[i] = worm.y

        accel = np.random.uniform(camel.acceleration - 0.10, camel.acceleration + 0.10)
        ang_vel = np.random.uniform(camel.angular_velocity - 0.1, camel.angular_velocity + 0.1)
        test.updateAgents(stepsize, (accel, ang_vel))
        i+=1

    CamXs[i:] = CamXs[i-1]
    CamYs[i:] = CamYs[i-1]
    WormXs[i:] = WormXs[i-1]
    WormYs[i:] = WormYs[i-1]
    fig, ax = plt.subplots()
    ax.plot(CamXs, CamYs, color='red', label='Camel')
    ax.plot(WormXs, WormYs, color='blue', label='Worm')
    if camel.is_alive:
        marker = 'o'
    else:
        marker = 'x'
    label = ('Death Spot', 'End Spot')
    ax.plot(CamXs[-1], CamYs[-1], marker=marker, color='g', label=label[int(camel.is_alive)])
    plt.xlabel('Epoch')
    plt.ylabel('Distance')
    plt.title('Distance vs Epoch')
    plt.grid(True)
    plt.xlim(test.min_x, test.max_x)
    plt.ylim(test.min_y, test.max_y)
    plt.legend()
    plt.show()



