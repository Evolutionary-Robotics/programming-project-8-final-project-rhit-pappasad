import bodies
import environment as env
from network import NeuralNetwork
from evoalg import EvoAlgorithm as Ealg
import pygame
import numpy as np
import pandas as pd
import pickle
import concurrent.futures
from copy import deepcopy as cpy
import matplotlib.pyplot as plt
import sys
import imageio




class Simulation:
    history_labels = ['camel x', 'camel y', 'camel angle', 'camel speed', 'camel angular_velocity',
                      'camel acceleration', 'camel hydration',
                      'distance']

    def __init__(self, desert: env.Desert, bg=env.DESERT_COLOR, noise=0.01):
        self.desert = desert
        self.bg = bg
        self.window_size = desert.size
        self.running = False
        self.camel = None
        self.worm = None
        self.ealg = None
        self.net = None

        self.net_inputs = bodies.Camel.state_returns
        self.net_outputs = bodies.Camel.trainables

        self.history = pd.DataFrame(columns=self.history_labels)
        self.history['sandworm state'] = pd.Series(dtype='string')
        self.history['time'] = pd.Series(dtype='int')

        self.noisestd = noise

    def runSingleQuick(self, stepsize, max_time):
        cx, cy, ca = self.camel.x, self.camel.y, self.camel.angle
        desert = self.desert
        self.quickRun(stepsize, max_time, desert, cx, cy, ca)

    def quickRunV2(self, stepsize, max_time, desert, camel):
        if self.net is None:
            print("ERROR: Network not initialized, did you train?")
            return

        t = 0
        X = np.zeros(max_time)
        Y = np.zeros(max_time)
        while t < max_time and camel.is_alive and not camel.reached_water:
            inp = camel.state()
            out = self.net.forward(inp) + np.random.normal(0, self.noisestd)
            desert.updateAgents(stepsize, out, 1)
            X[t] = camel.x
            Y[t] = camel.y
            t += stepsize

        X[t:] = X[t-1]
        Y[t:] = Y[t-1]

        return X, Y

    def runManyAndPlot(self, name, stepsize, max_time, num, save=False, random=False):
        if not random:
            top_row = np.array([[x, 50] for x in np.linspace(50, self.desert.max_x - 50, num // 4, endpoint=False)])
            left_col = np.array([[50, y] for y in np.linspace(self.desert.max_y-50, 50, num//4, endpoint=False)])
            bottom_row = np.array([[x, self.desert.max_y - 50] for x in np.linspace(self.desert.max_x - 50, 50, num // 4, endpoint=False)])
            right_col = np.array([[self.desert.max_x-50, y] for y in np.linspace(50, self.desert.max_y-50, num//4, endpoint=False)])
            starts = np.concatenate([top_row, right_col, bottom_row, left_col], axis=0)
            px = self.desert.size[0] // 2
            py = self.desert.size[1] // 2
        else:
            starts = np.random.randint(self.desert.min_x, self.desert.max_x, (num, 2))
            buff = np.sqrt(env.Desert.def_pool_size[0]**2 + env.Desert.def_pool_size[1]**2)
            px = np.random.randint(self.desert.min_x + buff, self.desert.max_x - buff)
            py = np.random.randint(self.desert.min_y + buff, self.desert.max_y - buff)
            name+='RandomStarts'

        speeds = np.random.uniform(0.0, bodies.Camel.max_speed, size=num)
        angles = np.random.uniform(0.0, 2 * np.pi, size=num)

        Xs = np.zeros((num, max_time))
        Ys = np.zeros((num, max_time))

        self.desert.addPool(px, py)
        deserts = [cpy(self.desert) for _ in range(num)]
        camels = [bodies.Camel(starts[n][0], starts[n][1], angles[n], deserts[n], speed=speeds[n]) for n in range(num)]
        with concurrent.futures.ThreadPoolExecutor(max_workers=num) as executor:
            futures = [executor.submit(self.quickRunV2, stepsize, max_time, deserts[n], camels[n]) for n in range(num)]
            for i, future in enumerate(concurrent.futures.as_completed(futures)):
               Xs[i], Ys[i] = future.result()

        fig, ax = plt.subplots()
        color = (env.DESERT_COLOR[0]/255, env.DESERT_COLOR[1]/255, env.DESERT_COLOR[2]/255, 1.0)
        ax.set_facecolor(color)  # Background for the plot area (axes)
        for n in np.arange(num):
            ax.plot(Xs[n][0], Ys[n][0], 'ro')
            ax.plot(Xs[n].T, Ys[n].T)
            ax.plot(Xs[n][-1], Ys[n][-1], 'g*')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title('Multiple Camel Visualization')
        plt.xlim(self.desert.min_x, self.desert.max_x)
        plt.ylim(self.desert.min_y, self.desert.max_y)

        circ = plt.Circle((self.desert.pool.x, self.desert.pool.y), self.desert.pool.radius, color='blue')
        ax.add_patch(circ)
        if save:
            plt.savefig(name+'MultipleViz.png', dpi=300)
        plt.show()

    def quickRun(self, stepsize, max_time, desert, camelx, camely, camela):
        if self.net is None:
            print("ERROR: Network not initialized, did you train?")
            return

        desert.addPool(desert.size[0] // 2, desert.size[1] // 2)
        speed = np.random.uniform(0.0, bodies.Camel.max_speed)
        temp_cam = bodies.Camel(camelx, camely, camela, desert, speed=speed)
        t = 0
        while t < max_time and temp_cam.is_alive and not temp_cam.reached_water:
            inp = temp_cam.state()
            out = self.net.forward(inp) + np.random.normal(0, self.noisestd)
            desert.updateAgents(stepsize, out, 1)
            t += stepsize

        return t

    def runManySim(self, name, stepsize, max_time, num, radius: int, save=False):
        results = np.zeros(num)
        theta = np.linspace(0.0, 2*np.pi, num)
        angles = np.random.uniform(0.0, 2 * np.pi, num)
        X = self.desert.size[0]//2 + radius * np.cos(theta)
        Y = self.desert.size[1]//2 + radius * np.sin(theta)
        deserts = [cpy(self.desert) for _ in range(num)]
        with concurrent.futures.ThreadPoolExecutor(max_workers=num) as executor:
            futures = [executor.submit(self.quickRun, stepsize, max_time, deserts[n], X[n], Y[n], angles[n]) for n in range(num)]
            for i, future in enumerate(concurrent.futures.as_completed(futures)):
                result = future.result()
                results[i] = result

        fi, ax = plt.subplots()
        ax.bar(*np.unique(results, return_counts=True)[:2])
        ax.set_xlabel('Time to Reach Water')
        ax.set_ylabel('Num Camels')
        ax.set_title('Multiple Camel Times to Reach Water')
        if save:
            plt.savefig(name+'ManyScatter', dpi=300)
        plt.show()
        return results

    def runViz(self, name, stepsize, max_time=1000, fps=60, save=False, debug=False):
        if self.net is None and not debug:
            print("ERROR: Network not initialized, did you train?")
            return

        pygame.init()
        screen = pygame.display.set_mode(self.window_size)
        clock = pygame.time.Clock()
        font = pygame.font.SysFont('Arial', 20)
        i = 100
        frames = []

        self.desert.addPool(self.desert.size[0]//2, self.desert.size[1]//2)
        self.addCamel()
        self.desert.pool.x = 600#np.random.randint(self.desert.pool.radius, self.desert.size[0] - self.desert.pool.radius)
        self.desert.pool.y = 600#np.random.randint(self.desert.pool.radius, self.desert.size[1] - self.desert.pool.radius)
        self.camel.x = 100
        self.camel.y = 100
        self.camel.angle = 0.0

        cooldown = [10, 10, 10]
        #self.addWorm()
        self.running = True
        t = 0
        camel_path = []
        worm_path = []
        while self.running and t < max_time:
            direction, forward = 0, 0
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                if event.type == pygame.KEYDOWN and debug:
                    direction, forward = 0, 0
                    if event.key == pygame.K_a:
                        direction = -1.0
                    elif event.key == pygame.K_w:
                        forward = 1.0
                    elif event.key == pygame.K_s:
                        forward = -1.0
                    elif event.key == pygame.K_d:
                        direction = 1.0

            if isinstance(self.bg, tuple):
                screen.fill(self.bg)

            #Update agents
            if self.camel is not None:
                if debug:
                    self.camel.handleManual(stepsize, forward, direction)
                elif self.camel.is_alive and not self.camel.reached_water:
                    inp = self.camel.state()
                    out = self.net.forward(inp) + np.random.normal(0, self.noisestd)
                    self.desert.updateAgents(stepsize, out, 1)
                    self.logHistory(stepsize)
                    camel_path.append((self.camel.x, self.camel.y))
                    if self.worm is not None:
                        worm_path.append((self.worm.x, self.worm.y))
                    text = font.render(f"{t} Eons", True, (0, 0, 0))
                    t_x = self.desert.size[0] // 2 - text.get_size()[0] // 2
                    t_y = 0
                    screen.blit(text, (t_x, t_y))
                else:
                    i -= 1
                    if i <= 0:
                        self.running = False

            if len(camel_path) > 1:
                pygame.draw.lines(screen, (255, 0, 0), False, camel_path, 2)

            if len(worm_path) > 1:
                pygame.draw.lines(screen, (155, 0, 155), False, worm_path, 5)

            if debug:
                state = np.round(self.camel.state(), 2)
                string = f"Angle: {state[0]} | Speed: {state[1]} | SeeWater?: {state[2]} | Distance: {state[3]} | Direction: {state[4]}"
                new_text = font.render(string, True, (0, 0, 0))
                tx = self.desert.size[0] // 2 - new_text.get_size()[0] // 2
                ty = 0
                screen.blit(new_text, (tx, ty))

            if self.worm is not None:
                self.drawBody(screen, self.worm)
            if self.camel is not None:
                self.drawBody(screen, self.camel)
            if self.desert.pool is not None:
                self.drawBody(screen, self.desert.pool)

            frame = pygame.surfarray.array3d(screen)
            frame = np.transpose(frame, (1, 0, 2))
            frames.append(frame)

            pygame.display.flip()
            clock.tick(fps)
            t+=stepsize

        pygame.quit()
        if save:
            frames = [frames[i] for i in range(len(frames)) if i % 2 == 0]
            print("Saving gif...")
            imageio.mimsave(name+'.gif', frames, fps=2*fps)
            print("Saved.")

    def trainCamel(self, popsize, stepsize, max_time, hidden, trials_per=3, name='Train', fast=True):
        print("Training", name, "...")
        fitness_func = self.createFitnessFunction(stepsize, max_time, hidden, trials_per, fast)

        layers = [self.net_inputs] + hidden + [self.net_outputs]
        genesize = np.sum(np.multiply(layers[1:], layers[:-1]))
        r_prob = 0.9
        m_prob = 0.2
        tourney = popsize * 50
        self.ealg = Ealg(fitness_func, genesize, popsize, r_prob, m_prob, tourney)
        self.ealg.runAlg(True)
        self.ealg.plot(name, True)

        history = self.ealg.getHistory()
        best_idx = int(history['Best Idx'].iloc[-1])
        best_genes = self.ealg.population[best_idx]
        self.net = NeuralNetwork(self.net_inputs, hidden, self.net_outputs, 'relu', 'sigmoid')
        self.net.setParams(best_genes)
        self.save(best_genes, name)

    def save(self, genes, name):
        path = f'{name}.pkl'
        tup = (self.net.hidden_map, genes)
        with open(path, 'wb') as f:
            pickle.dump(tup, f)

        print("Saved data into:", path)
        return True

    def load(self, path):
        with open(path, 'rb') as f:
            tup = pickle.load(f)

        hidden, best_genes = tup
        self.net = NeuralNetwork(self.net_inputs, hidden, self.net_outputs, 'relu', 'sigmoid')
        self.net.setParams(best_genes)

        print("Loaded data from:", path)
        return True

    def createFitnessFunction(self, stepsize, max_time, hidden, trials_per, fast: bool):
        desert = self.desert
        x_range = np.linspace(desert.min_x, desert.max_x, trials_per)
        y_range = np.linspace(desert.min_y, desert.max_y, trials_per)
        a_range = np.linspace(0, 2*np.pi, trials_per)
        s_range = np.linspace(0, bodies.Camel.max_speed, trials_per)
        poolx_range = np.linspace(desert.min_x + desert.def_pool_size[0], desert.max_x - desert.def_pool_size[0], trials_per)
        pooly_range = np.linspace(desert.min_y + desert.def_pool_size[1], desert.max_y - desert.def_pool_size[1], trials_per)
        total_trials = trials_per**6
        camel = bodies.Camel(0, 0, 0, desert)
        desert.addPool(0, 0)

        def daLoop(net, distance, desert, camel):
            fit = 0.0
            t = 0
            f = stepsize
            while t < max_time and f > 0.0:
                inp = camel.state()
                out = net.forward(inp) + np.random.normal(0.0, self.noisestd)
                f = desert.updateAgents(stepsize, out, distance)
                fit += f
                t += stepsize

                if f == stepsize:
                    fit += (max_time - t)
                    t += (max_time - t)

            return fit

        def fast_fitness_func(genotype):
            net = NeuralNetwork(self.net_inputs, hidden, self.net_outputs, 'relu', 'sigmoid')
            net.setParams(genotype)
            fits = np.zeros(total_trials)
            deserts = [cpy(desert) for _ in range(total_trials)]
            for _ in range(total_trials): desert.addPool(0, 0)
            camels = [bodies.Camel(0, 0, 0, deserts[n]) for n in range(total_trials)]
            distances = np.zeros(total_trials)

            i = 0
            for px in poolx_range:
                for py in pooly_range:
                    for x in x_range:
                        for y in y_range:
                            for a in a_range:
                                for s in s_range:
                                    deserts[i].pool.x = px
                                    deserts[i].pool.y = py
                                    camels[i].reset()
                                    camels[i].x = x
                                    camels[i].y = y
                                    camels[i].angle = a
                                    camels[i].speed = s
                                    distances[i] = camels[i].getDistance(px, py) - deserts[i].pool.radius
                                    i+=1

            with concurrent.futures.ThreadPoolExecutor(max_workers=total_trials) as executor:
                futures = [executor.submit(daLoop, net, distances[n], deserts[n], camels[n]) for n in range(total_trials)]

                for i, future in enumerate(concurrent.futures.as_completed(futures)):
                    fits[i] = future.result()

            fit = np.sum(fits)
            return fit / (max_time * total_trials)

        def fitness_func(genotype):
            net = NeuralNetwork(self.net_inputs, hidden, self.net_outputs, 'relu', 'sigmoid')
            net.setParams(genotype)
            fit = 0.0
            for px in poolx_range:
                for py in pooly_range:
                    for x in x_range:
                        for y in y_range:
                            for a in a_range:
                                for s in s_range:
                                    camel.reset()
                                    camel.x = x
                                    camel.y = y
                                    camel.angle = a
                                    camel.speed = s
                                    desert.pool.x = px
                                    desert.pool.y = py
                                    max_distance = camel.getDistance(px, py) - desert.pool.radius

                                    f = stepsize
                                    t = 0
                                    while t < max_time and f > 0.0:
                                        inp = camel.state()
                                        out = net.forward(inp) + np.random.normal(0.0, self.noisestd)
                                        f = desert.updateAgents(stepsize, out, max_distance)
                                        fit += f
                                        t += stepsize

                                        if f == stepsize:
                                            fit += (max_time - t)
                                            t += (max_time - t)

            return fit / (max_time * total_trials)

        return fitness_func if not fast else fast_fitness_func

    def logHistory(self, stepsize):
        new_entry = pd.Series(index=self.history_labels)
        for label in self.history.columns:
            words = label.split()
            if words[0] in ('camel'):
                new_entry[label] = self.__getattribute__(words[0]).__getattribute__(words[1])
            elif words[0] == 'distance':
                new_entry[label] = self.camel.getDistance(self.desert.pool.x - self.desert.pool.radius, self.desert.pool.y - self.desert.pool.radius)
            elif words[0] == 'time':
                new_entry[label] = self.history.shape[0] * stepsize
            else:
                pass
                # rint(f"ERROR: Unable to parse metric -> {label}")
                # exit(1)


        self.history.loc[self.history.shape[0]] = new_entry

    def generateCoords(self, buffer=50):
        x = np.random.randint(self.desert.min_x + buffer, self.desert.max_x - buffer)
        y = np.random.randint(self.desert.min_y + buffer, self.desert.max_y - buffer)
        direction = np.random.uniform(-np.pi, np.pi)
        return x, y, direction

    def addCamel(self, camel=None, add_worm=False):
        if camel is None:
            x, y, direction = self.generateCoords()
            x = np.random.choice([0.1*self.desert.max_x, 0.9*self.desert.max_x])
            speed = np.random.uniform(0.0, bodies.Camel.max_speed)
            self.camel = bodies.Camel(x, y, direction, self.desert, speed=speed)
        else:
            self.camel = camel

        if add_worm:
            self.addWorm()

    def addWorm(self, worm=None):
        if worm is None:
            if self.camel is None:
                print("ERROR: Add Camel Before Worm")
                exit(1)

            x, y, direction = self.generateCoords()
            self.worm = bodies.Sandworm(x, y, direction, self.desert)
            while self.worm.getDistance(self.camel.x, self.camel.y) <= self.worm.detection_radius * 2:
                x, y, direction = self.generateCoords()
                self.worm = bodies.Sandworm(x, y, direction, self.desert)
        else:
            if self.camel is None:
                print("ERROR: Add Camel Before Worm")
                exit(1)

            self.worm = worm

    def drawArrow(self, screen, arrow):
        color = arrow[1]
        lines = arrow[2:]
        for line in lines:
            pygame.draw.line(screen, color, *line)

    def drawBody(self, screen, body):
        if isinstance(body, bodies.Camel):
            self.drawArrow(screen, body.getArrow())

        body_info = body.getShape()
        shape = body_info[0]
        if shape == 'Circle':
            pygame.draw.circle(screen, *body_info[1:])
        elif shape == 'X':
            l1 = body_info[1]
            l2 = body_info[2]
            pygame.draw.line(screen, *l1)
            pygame.draw.line(screen, *l2)

    def plot(self, name, save=False):
        plottable = ['distance', 'camel speed', 'camel acceleration', 'camel angular_velocity', 'camel angle']
        plottable = self.history[plottable]
        dist = plottable[['distance']]
        ang = plottable[['camel angle', 'camel angular_velocity']]
        sped = plottable[['camel speed', 'camel acceleration']]

        plot = dist.plot(grid=True, xlabel='Time', title='Distance From Pool Over Time')
        if save:
            plt.savefig(name+'Distance.png', dpi=300)
        plt.show()

        plot = ang.plot(grid=True, xlabel='Time', title='Angle Over Time')
        if save:
            plt.savefig(name+'Angle.png', dpi=300)
        plt.show()

        plot = sped.plot(grid=True, xlabel='Time', title='Speed Over Time')
        if save:
            plt.savefig(name+'Speed.png', dpi=300)
        plt.show()


def test_moderate_big():
    desert = env.Desert()
    sim = Simulation(desert)
    stepsize = 1
    popsize = 10
    max_time = 800
    hidden = [2, 4, 8, 16, 8, 4, 2]
    sim.trainCamel(popsize, stepsize, max_time, hidden, name='Big', fast=True)
    sim.runViz(stepsize)
    sim.plot(True)

def test_thicc():
    desert = env.Desert()
    sim = Simulation(desert)
    stepsize = 1
    popsize = 10
    max_time = 800
    hidden = [4, 4, 4, 4]
    sim.trainCamel(popsize, stepsize, max_time, hidden, name='Thicc', fast=True)
    sim.runViz(stepsize)
    sim.plot(True)

def test_mid():
    desert = env.Desert()
    sim = Simulation(desert)
    stepsize = 1
    popsize = 10
    max_time = 1000
    hidden = [8, 16, 8, 4]
    sim.trainCamel(popsize, stepsize, max_time, hidden, name='Mid', trials_per=4, fast=True)
    sim.runViz('h', stepsize)
    sim.plot(True)

def test_simple():
    print("Running test for Simulation class...")
    desert = env.Desert()
    sim = Simulation(desert)
    popsize = 10
    stepsize = 1
    max_time = 1000
    hidden = [5]
    print("Training the camel...")
    sim.trainCamel(popsize=popsize, stepsize=stepsize, max_time=max_time, hidden=hidden)
    sim.runViz('simple', stepsize)
    sim.plot(True)

def test_plot_simple(name):
    path = f'Saved/{name}.pkl'
    desert = env.Desert()
    sim = Simulation(desert)
    sim.load(path)
    stepsize = 1
    max_time = 1000
    sim.addCamel()
    sim.runManyAndPlot(name, stepsize, max_time, 20, save=True, random=True)
    #sim.plot(True)

def test_many_simple(name):
    print("Running many tests for Simulation class...")
    desert = env.Desert()
    sim = Simulation(desert)
    x = desert.size[0] * 0.1
    y = desert.size[1] * 0.1
    camel = bodies.Camel(x, y, 0, desert)
    path = f'Saved/{name}.pkl'
    sim.load(path)
    stepsize = 1
    max_time = 1000
    num = 1000
    distance = 300
    results = sim.runManySim(name, stepsize, max_time, num, distance, save=True)
    sim.runManyAndPlot(name, stepsize, max_time, 20, save=True, random=False)

def testFromSaved(name):
    path = f'Saved/{name}.pkl'
    desert = env.Desert()
    sim = Simulation(desert)
    sim.load(path)
    stepsize = 1
    sim.runViz(name, stepsize, 1000, save=True)
    #sim.runManyAndPlot(name, stepsize, 400, 20, save=True, random=False)

def showNet(name):
    path = f'Saved/{name}.pkl'
    desert = env.Desert()
    sim = Simulation(desert)
    sim.load(path)
    sim.net.visualize(name, 1)



funcs = [
    test_simple,
    test_mid,
    test_thicc,
    test_moderate_big
]

def debugCamel():
    desert = env.Desert()
    sim = Simulation(desert)
    stepsize = 1
    max_time = 100000
    sim.runViz('debug', stepsize, max_time, debug=True)

if __name__ == '__main__':
     #i = int(sys.argv[1])
     #funcs[i]()
     #funcs[0]()
     #showNet('Train')
     # showNet('Mid')
     # showNet('Thicc')
     # showNet('Big')
     #test_mid()
     #debugCamel()
     #test_mid()
     testFromSaved('Mid')
     #test_simple()
     #test_plot_simple('Mid')
     #showNet('Mid')