
import pandas as pd
import numpy as np
from MaxCut import MaxCut
from QAOA import QAOA
from EDAspy.optimization.univariate import EDA_continuous as EDAc
from qiskit import Aer
import random
import pickle

random.seed(1234)

n_nodes = 10
p = 2
n_shots = 100

max_cut = MaxCut(n_nodes=n_nodes)
max_cut.random_graph(per_arcs=0.8)

with open('max_cut_'+str(n_nodes)+'.pkl', 'wb') as config_max_cut_file:
    pickle.dump(max_cut, config_max_cut_file)

# initial vector of statistics
vector = pd.DataFrame(columns=list(range(0, p*2)))
vector['data'] = ['mu', 'std', 'min', 'max']
vector = vector.set_index('data')
vector.loc['mu'] = np.pi
vector.loc['std'] = 0.2

vector.loc['min'] = 0  # optional
vector.loc['max'] = np.pi*2  # optional

qaoa = QAOA(max_cut=max_cut, p=p, backend=Aer.get_backend('qasm_simulator'))

EDA = EDAc(SIZE_GEN=10, MAX_ITER=5, DEAD_ITER=3, ALPHA=0.7, vector=vector,
           aim='minimize', cost_function=qaoa.compute_minimum_eigenvalue(n_shots=n_shots))

best_cost, params, history = EDA.run(output=True)

print('Best solution:', params.to_dict())
print('Best cost:', best_cost)
print('Evolution:', history)
