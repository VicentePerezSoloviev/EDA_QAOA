import pandas as pd
import numpy as np
from qiskit.algorithms import QAOA
from continuous_qiskit_optimized import UMDAc as EDAc
from qiskit import Aer
import random
import time
from qiskit.utils import QuantumInstance
from qubo_formulation import QUBO
from scores import Scores
from qiskit.quantum_info.operators import Operator

random.seed(1234)

# qubit_op, _ = max_cut.get_operator()

scores = Scores()
weights = scores.load_data("cancer.txt")
qubo = QUBO(scores, m=2)
Q = qubo.q(17000, 17000, 17000)
qubit_op = Operator(qubo.Q)

def create_vector(p):
    vec = pd.DataFrame(columns=list(range(0, p * 2)))
    vec['data'] = ['mu', 'std', 'min', 'max']
    vec = vec.set_index('data')
    vec.loc['mu'] = np.pi
    vec.loc['std'] = 0.5

    vec.loc['min'] = 0  # optional
    vec.loc['max'] = np.pi * 2  # optional
    return vec


ps = range(1, 12)
size_gens = [10, 20, 30]
size_gen = 20
p = 3
vector = create_vector(p)
eda = EDAc(size_gen=size_gen, max_iter=100, dead_iter=20, alpha=0.7, vector=vector)

counts = []
values = []


def store_intermediate_result(eval_count, parameters, mean, std):
    counts.append(eval_count)
    values.append(mean)


vqe = QAOA(eda, callback=store_intermediate_result,
           quantum_instance=QuantumInstance(backend=Aer.get_backend('statevector_simulator')), reps=p)
# by default 1024 shots
start_time = time.process_time()
result = vqe.compute_minimum_eigenvalue(operator=qubit_op)  # result is VQEResult
finish_time = time.process_time()

print('eda', p, size_gen, result.optimal_value, finish_time-start_time)

