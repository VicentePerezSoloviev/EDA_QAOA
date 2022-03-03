import pandas as pd
import numpy as np
from qiskit.algorithms import QAOA
from continuous_qiskit import UMDAc as EDAc
from qiskit import Aer
import random
import time
import pickle
from qiskit.utils import QuantumInstance

random.seed(1234)


with open('max_cut_10.pkl', 'rb') as file:
    max_cut = pickle.load(file)

qubit_op, _ = max_cut.get_operator()


def create_vector(p):
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

filename = 'output_eda.csv'
dt = pd.DataFrame(columns=['opt', 'it', 'p', 'size_gen', 'best_cost', 'time'])

index = 0
for p in ps:
    for size_gen in size_gens:
        for it in range(10):
            vector = create_vector(p)
            eda = EDAc(size_gen=size_gen, max_iter=200, dead_iter=15, alpha=0.7, vector=vector)

            counts = []
            values = []

            def store_intermediate_result(eval_count, parameters, mean, std):
                counts.append(eval_count)
                values.append(mean)

            vqe = QAOA(eda, callback=store_intermediate_result,
                       quantum_instance=QuantumInstance(backend=Aer.get_backend('statevector_simulator')), reps=p)
            start_time = time.time()
            result = vqe.compute_minimum_eigenvalue(operator=qubit_op)
            finish_time = time.time()

            dt.loc[index] = ['eda', it, p, size_gen, min(values), finish_time - start_time]
            dt.to_csv(filename)
            index = index + 1

            # TODO: save optimum parameters
