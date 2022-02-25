
import pandas as pd
import numpy as np
from MaxCut import MaxCut
from QAOA import QAOA
from EDAspy.optimization.univariate import EDA_continuous as EDAc
from qiskit import Aer
import random
import time
import pickle
import matplotlib.pyplot as plt

random.seed(1234)

n_nodes = 10
p = 2
n_shots = 100

# max_cut = MaxCut(n_nodes=n_nodes)
# max_cut.random_graph(per_arcs=0.8)

# with open('max_cut_'+str(n_nodes)+'.pkl', 'wb') as config_max_cut_file:
#     pickle.dump(max_cut, config_max_cut_file)

with open('max_cut_10.pkl', 'rb') as file:
    max_cut = pickle.load(file)

# initial vector of statistics
vector = pd.DataFrame(columns=list(range(0, p*2)))
vector['data'] = ['mu', 'std', 'min', 'max']
vector = vector.set_index('data')
vector.loc['mu'] = np.pi
vector.loc['std'] = 0.5

vector.loc['min'] = 0  # optional
vector.loc['max'] = np.pi*2  # optional

qaoa = QAOA(max_cut=max_cut, p=p, backend=Aer.get_backend('qasm_simulator'))
plt.figure(figsize=(6, 6))
for i in [2, 5, 7, 10, 15]:
    print(i)
    history = []
    start_time = time.time()
    EDA = EDAc(SIZE_GEN=i, MAX_ITER=50, DEAD_ITER=10, ALPHA=0.7, vector=vector,
               aim='minimize', cost_function=qaoa.compute_minimum_eigenvalue(n_shots=n_shots))

    best_cost, params, history = EDA.run(output=False)
    print('time:', time.time() - start_time)
    print('Best solution:', params.to_dict())
    print('Best cost:', best_cost)
    print('Evolution:', history)

    plt.plot(range(len(history)), history)

plt.savefig('eda.png')
