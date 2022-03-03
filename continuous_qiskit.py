import pandas as pd
import numpy as np
from scipy.stats import norm

'''
In this version of UMDA, instead of a vector of probabilities, a vector of univariate normal distributions is found
When sampling, it is sampled from gaussian
vector is a table with, columns as variables, and rows with mu, std, and optional max and min
'''


class UMDAc:

    """Univariate marginal Estimation of Distribution algorithm continuous.
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
    :param aim: Represents the optimization aim.
    :type aim: 'minimize' or 'maximize'.
    :param cost_function: a callable function implemented by the user, to optimize.
    :type cost_function: callable function which receives a dictionary as input and returns a numeric

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
        if dead_iter >= max_iter:
            raise Exception('ERROR setting DEAD_ITER. The dead iterations must be fewer than the maximum iterations')
        else:
            self.DEAD_ITER = dead_iter

    def set_max_evals_grouped(self, max_evals_grouped):
        pass

    # new individual
    def __new_individual__(self):
        """Sample a new individual from the vector of probabilities.
        :return: a dictionary with the new individual; with names of the parameters as keys and the values.
        :rtype: dict
        """

        dic = {}
        for var in self.variables:
            mu = self.vector.loc['mu', var]
            std = self.vector.loc['std', var]
            sample = np.random.normal(mu, std, 1)
            while sample < self.vector.loc['min', var] or sample > self.vector.loc['max', var]:
                sample = np.random.normal(mu, std, 1)

            dic[var] = sample[0]
        return dic

    # build a generation of size SIZE_GEN from prob vector
    def new_generation(self):
        """Build a new generation sampled from the vector of probabilities. Updates the generation pandas dataframe
        """
        gen = pd.DataFrame(columns=self.variables)
        while len(gen) < self.SIZE_GEN:
            individual = self.__new_individual__()
            gen = gen.append(individual, True)

            # drop duplicate individuals
            gen = gen.drop_duplicates()
            gen = gen.reset_index(drop=True)

        # self.generation = gen
        if type(self.generation) is pd.DataFrame:
            self.generation = self.generation.nsmallest(int(self.elite_factor*len(self.generation)), 'cost')
            self.generation = self.generation.append(gen).reset_index(drop=True)
        else:
            self.generation = gen

    # truncate the generation at alpha percent
    def truncation(self):
        """ Selection of the best individuals of the actual generation. Updates the generation by selecting the best individuals
        """
        length = int(self.SIZE_GEN * self.alpha)
        self.generation = self.generation.nsmallest(length, 'cost')

    # check each individual of the generation
    def check_generation(self, objective_function):
        """Check the cost of each individual in the cost function implemented by the user
        """
        for ind in range(len(self.generation)):
            cost = objective_function(self.generation.loc[ind, self.generation.columns != 'cost'].to_list())
            self.generation.loc[ind, 'cost'] = cost

    # update the probability vector
    def update_vector(self):
        """From the best individuals update the vector of normal distributions in order to the next
        generation can sample from it. Update the vector of normal distributions
        """

        for var in self.variables:
            array = self.generation[var].values

            # calculate mu and std from data
            mu, std = norm.fit(array)

            # std should never be 0
            if std < 0.3:
                std = 0.3

            # update the vector probabilities
            self.vector.loc['mu', var] = mu
            self.vector.loc['std', var] = std

    # intern function to compare local cost with global one
    def __compare_costs__(self, local):
        """Check if the local best cost is better than the global one
        :param local: local best cost
        :type local: float
        :return: True if is better, False if not
        :rtype: bool
        """

        return local < self.best_mae_global

    def minimize(self, fun, x0, jac, bounds):

        not_better = 0
        for i in range(self.MAX_ITER):
            self.new_generation()
            self.check_generation(fun)
            self.truncation()
            self.update_vector()

            best_mae_local = self.generation['cost'].min()

            self.history.append(best_mae_local)
            best_ind_local = self.generation[self.generation['cost'] == best_mae_local]
            print(i, self.best_mae_global, not_better, best_mae_local)

            # update the best values ever
            # if best_mae_local <= self.best_mae_global:
            if self.__compare_costs__(best_mae_local):
                self.best_mae_global = best_mae_local
                self.best_ind_global = best_ind_local
                not_better = 0
            else:
                not_better = not_better + 1
                if not_better == self.DEAD_ITER:
                    return EdaResult(self.best_ind_global.reset_index(drop=True).loc[0].to_list()[:-1], self.best_mae_global, len(self.history))

        # return self.best_mae_global, self.best_ind_global, self.history
        return EdaResult(self.best_ind_global.reset_index(drop=True).loc[0].to_list()[:-1], self.best_mae_global, len(self.history))


class EdaResult:
    def __init__(self, optimal_point, optimal_value, cost_function_evals):
        self.x = optimal_point
        self.fun = optimal_value
        self.nfev = cost_function_evals
