import pandas as pd
import numpy as np
from scipy.stats import norm

'''
In this version of UMDA, instead of a vector of probabilities, a vector of univariate normal distributions is found
When sampling, it is sampled from gaussian
vector is a table with, columns as variables, and rows with mu, std, and optional max and min
'''


class UMDAc:

    """Continuous univariate marginal Estimation of Distribution algorithm.
    New individuals are sampled from a vector of univariate normal distributions.

    :param size_gen: total size of the generations in the execution of the algorithm
    :type size_gen: int
    :param max_iter: total number of iterations in case that optimum is not yet found. If reached, the optimum found is returned
    :type max_iter: int
    :param dead_iter: total number of iteration with no better solution found. If reached, the optimum found is returned
    :type dead_iter: int
    :param alpha: percentage of the generation tu take, in order to sample from them. The best individuals selection
    :type alpha: float [0-1]
    :param vector: vector of normal distributions to sample from
    :type vector: pandas dataframe with columns ['mu', 'std'] and optional ['min', 'max']

    :raises Exception: cost function is not callable

    """

    SIZE_GEN = -1
    MAX_ITER = -1
    DEAD_ITER = -1
    alpha = -1
    vector = -1

    generation = -1

    best_mae_global = -1
    best_ind_global = -1

    cost_function = -1
    history = []
    setting = "---"

    elite_factor = 0.4

    def __init__(self, size_gen, max_iter, dead_iter, alpha, vector):
        """Constructor of the optimizer class
        """

        self.SIZE_GEN = size_gen
        self.MAX_ITER = max_iter
        self.alpha = alpha
        self.vector = vector

        self.variables = list(vector.columns)

        self.best_mae_global = 999999999999

        # self.DEAD_ITER must be fewer than MAX_ITER
        if dead_iter > max_iter:
            raise Exception('ERROR setting DEAD_ITER. The dead iterations must be fewer than the maximum iterations')
        else:
            self.DEAD_ITER = dead_iter

        self.truncation_length = int(size_gen * alpha)

        # initialization of generation
        mus = self.vector[self.variables].loc['mu'].to_list()
        stds = self.vector[self.variables].loc['std'].to_list()
        self.generation = pd.DataFrame(np.random.normal(mus, stds, [self.SIZE_GEN, len(self.variables)]),
                                       columns=self.variables, dtype='float_')

    def set_max_evals_grouped(self, max_evals_grouped):
        pass

    # build a generation of size SIZE_GEN from prob vector
    def new_generation(self):
        """Build a new generation sampled from the vector of probabilities. Updates the generation pandas dataframe
        """

        mus = self.vector[self.variables].loc['mu'].to_list()
        stds = self.vector[self.variables].loc['std'].to_list()
        gen = pd.DataFrame(np.random.normal(mus, stds, [self.SIZE_GEN, len(self.variables)]),
                           columns=self.variables, dtype='float_')

        self.generation = self.generation.nsmallest(int(self.elite_factor*len(self.generation)), 'cost')
        self.generation = self.generation.append(gen).reset_index(drop=True)

    # truncate the generation at alpha percent
    def truncation(self):
        """Selection of the best individuals of the actual generation. Updates the generation by selecting the best individuals
        """
        self.generation = self.generation.nsmallest(self.truncation_length, 'cost')

    # check each individual of the generation
    def check_generation(self, objective_function):
        """Check the cost of each individual in the cost function implemented by the user
        """
        self.generation['cost'] = self.generation.apply(lambda row: objective_function(row[self.variables].to_list()),
                                                        axis=1)

    # update the probability vector
    def update_vector(self):
        """From the best individuals update the vector of normal distributions in order to the next
        generation can sample from it. Update the vector of normal distributions
        """

        for var in self.variables:
            self.vector.at['mu', var], self.vector.at['std', var] = norm.fit(self.generation[var].values)
            if self.vector.at['std', var] < 0.3:
                self.vector.at['std', var] = 0.3

    def minimize(self, fun, x0, jac, bounds):

        not_better = 0
        for i in range(self.MAX_ITER):
            self.check_generation(fun)
            self.truncation()
            self.update_vector()

            best_mae_local = self.generation['cost'].min()

            self.history.append(best_mae_local)
            best_ind_local = self.generation[self.generation['cost'] == best_mae_local]

            # update the best values ever
            if best_mae_local < self.best_mae_global:
                self.best_mae_global = best_mae_local
                self.best_ind_global = best_ind_local
                not_better = 0
            else:
                not_better += 1
                if not_better == self.DEAD_ITER:
                    return EdaResult(self.best_ind_global.reset_index(drop=True).loc[0].to_list()[:-1], self.best_mae_global, len(self.history))

            self.new_generation()

        return EdaResult(self.best_ind_global.reset_index(drop=True).loc[0].to_list()[:-1], self.best_mae_global, len(self.history))


class EdaResult:
    def __init__(self, optimal_point, optimal_value, cost_function_evals):
        self.x = optimal_point
        self.fun = optimal_value
        self.nfev = cost_function_evals
