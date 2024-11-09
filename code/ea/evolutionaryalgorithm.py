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
            raise ValueError("Num tourney should be greater than or equal to pop size.")

        self.fitness_func = fitness_func
        self.gene_size = num_genes
        self.r_prob = recombination_rate
        self.m_prob = mutation_rate
        self.num_tourney = num_tourney

        self.pop_size = pop_size
        self.population = np.random.uniform(-1, 1, (self.pop_size, self.gene_size))
        self.fits = self.get_fitness()

        self.num_generations = self.num_tourney // self.pop_size
        self.history = np.zeros((self.num_generations, len(self.MAP)), dtype=np.float32)

    def get_fitness(self):
        return np.apply_along_axis(self.fitness_func, 1, self.population)

    def run_algorithm(self, do_print=False):
        generation = 0
        for t in range(self.num_tourney):
            # Select two individuals to compete using vectorized random choice
            indices = np.random.choice(self.pop_size, 2, replace=False)
            winner, loser = indices[np.argsort(self.fits[indices])[::-1]]

            # Recombine winner's genes to the loser based on recombination probability
            recombination_mask = np.random.rand(self.gene_size) < self.r_prob
            self.population[loser] = np.where(recombination_mask, self.population[winner], self.population[loser])

            # Apply mutation to the loser's genes
            mutation_values = np.random.normal(0, self.m_prob, self.gene_size)
            self.population[loser] += mutation_values
            np.clip(self.population[loser], -1, 1, out=self.population[loser])

            # Update the loser's fitness
            self.fits[loser] = self.fitness_func(self.population[loser])

            # Record statistics at the end of each generation
            if t % self.pop_size == 0:
                for idx, (_, func) in enumerate(self.MAP.items()):
                    self.history[generation, idx] = func(self.fits)
                if do_print:
                    print(f"Generation {generation}: " +
                          " | ".join([f"{label}: {val:.4f}" for label, val in zip(self.MAP.keys(), self.history[generation])]))
                generation += 1

        if do_print:
            print("Algorithm completed.")

    def get_history(self):
        return pd.DataFrame(self.history, columns=list(self.MAP.keys()))

    def plot(self, title, save=False):
        df = self.get_history()
        df.loc[:, [col for col in df.columns if 'idx' not in col.lower()]].plot(
            kind='line', grid=True, title=title, xlabel='Generation', ylabel='Fitness'
        )
        if save:
            plt.savefig(f'{title}.png', bbox_inches='tight', dpi=300)
        plt.show()

if __name__ == '__main__':
    genesize = 5
    popsize = 100
    num_tourney = popsize * 10
    fitness_func = lambda x: np.sum(x)
    recombination_rate = 0.7
    mutation_rate = 0.01
    title = 'Optimized EA'

    ea = EvoAlgorithm(fitness_func, genesize, popsize, recombination_rate, mutation_rate, num_tourney)
    ea.plot(title + ' Before Tourney')
    ea.run_algorithm(do_print=True)
    ea.plot(title + ' After Tourney')
    print(ea.get_history())
