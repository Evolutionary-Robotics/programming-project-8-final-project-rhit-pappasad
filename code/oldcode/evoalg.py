import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class EvoAlgorithm:
    MAP = {
        'Best Fit': np.max,
        'Avg Fit': np.mean,
        'Worst Fit': np.min,
        'Best Idx': np.argmax,
        'Worst Idx': np.argmin
    }

    def __init__(self, fitness_func, num_genes, pop_size, recombination_rate, mutation_rate, num_tourney):
        if num_tourney < pop_size:
            print("ERROR: Num tourney is less than pop size")
            exit(1)

        self.fitness_func = fitness_func
        self.gene_size = num_genes
        self.r_prob = recombination_rate
        self.m_prob = mutation_rate
        self.num_tourney = num_tourney

        self.pop_size = pop_size
        self.population = np.random.uniform(-1, 1, (self.pop_size, self.gene_size))
        self.pop_idx = np.indices(self.population.shape)[0].flatten()

        self.num_generations = int(self.num_tourney // self.pop_size)
        self.history = np.zeros((self.num_generations, len(self.MAP))).astype(np.float32)

        self.fits = self.getFitness()


    def getFitness(self):
        return np.apply_along_axis(self.fitness_func, 1, self.population)

    def runAlg(self, do_print=False):
        generation = 0
        for t in np.arange(self.num_tourney):
            #Pick two to fight
            a, b = np.random.choice(self.pop_idx, 2, replace=False)
            #Pick winner
            w_idx = max(a, b, key=lambda i: self.fits[i])
            l_idx = min(a, b, key=lambda i: self.fits[i])
            #transfect winner to loser
            for g in np.arange(self.gene_size):
                if np.random.random() < self.r_prob:
                    self.population[l_idx][g] = self.population[w_idx][g]
            #mutate loser
            self.population[l_idx] += np.random.normal(0, self.m_prob, self.gene_size)
            self.population[l_idx] = np.clip(self.population[l_idx], -1, 1)
            #update
            self.fits[l_idx] = self.fitness_func(self.population[l_idx])
            #stats
            if t % self.pop_size == 0:
                if do_print: s = f"{t}: "
                for idx, (label, function) in enumerate(self.MAP.items()):
                    val = function(self.fits)
                    self.history[generation][idx] = val
                    if do_print: s += f"{label}: {np.float32(val)} | "
                if do_print:
                    s = s[:-2]
                    print(s)
                generation += 1
        if do_print:
            print("DONE RUN")

    def getHistory(self):
         return pd.DataFrame(self.history, columns=list(self.MAP.keys()))

    def plot(self, title, save=False):
        df = self.getHistory()
        plottable = df[[col for col in df.columns if 'idx' not in col.lower()]]
        plottable.plot(kind='line', grid=True, legend=True, title=title, xlabel='Generation', ylabel='Fitness')
        if save:
            plt.savefig('EvoAlg '+title+'.png', bbox_inches='tight', dpi=300)
        plt.show()





if __name__ == '__main__':
    genesize = 5
    popsize = 100
    t = popsize*10
    fitness_func = lambda x: np.sum(x)
    rr = 0.7
    mr = 0.01
    title = 'TestEA'

    ea = EvoAlgorithm(fitness_func, genesize, popsize, rr, mr, t)
    ea.plot(title + ' Before Tourney')
    ea.runAlg(do_print=True)
    ea.plot(title + ' After Tourney')
    print(ea.getHistory())
