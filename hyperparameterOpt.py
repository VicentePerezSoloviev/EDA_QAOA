import pandas as pd
import numpy as np
from QAOA import QAOA
from continuous import UMDAc as EDAc
from qiskit import Aer
import random
import time
import pickle

random.seed(1234)

# n_nodes = 10
# p = 2
# n_shots = 100

with open('max_cut_10.pkl', 'rb') as file:
    max_cut = pickle.load(file)


def create_vector():
    vec = pd.DataFrame(columns=list(range(0, p * 2)))
    vec['data'] = ['mu', 'std', 'min', 'max']
    vec = vec.set_index('data')
    vec.loc['mu'] = np.pi
    vec.loc['std'] = 0.5

    vec.loc['min'] = 0  # optional
    vec.loc['max'] = np.pi * 2  # optional
    return vec


ps = range(1, 8)
n_shotss = [50, 100, 200]
size_gens = [2, 5, 10, 15]

results = pd.DataFrame(columns=['p', 'n_shots', 'size_gen', 'it', 'time', 'best_cost', 'conv'])
filename = 'results.csv'

index = 0
for p in ps:
    for n_shots in n_shotss:
        for size_gen in size_gens:
            for it in range(10):
                qaoa = QAOA(max_cut=max_cut, p=p, backend=Aer.get_backend('qasm_simulator'))

                history = []
                vector = create_vector()

                EDA = EDAc(SIZE_GEN=size_gen, MAX_ITER=50, DEAD_ITER=10, ALPHA=0.7, vector=vector,
                           aim='minimize', cost_function=qaoa.compute_minimum_eigenvalue(n_shots=n_shots))

                start_time = time.time()
                best_cost, params, history = EDA.run()
                finish_time = time.time()

                results.loc[index] = [p, n_shots, size_gen, it,
                                      finish_time-start_time, best_cost, len(history)]
                index = index + 1
                results.to_csv(filename)

                # TODO: save optimum parameters
