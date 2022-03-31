from scipy.optimize import minimize
import pickle
from QAOA import QAOA
from qiskit import Aer
import random
import numpy as np

from qiskit import execute
from QAOA import compute_expectation_value
global iters


def random_init_parameters(layers):
    return np.array([random.uniform(0, np.pi) for i in range(layers*2)])


def check_result(x):
    qaoa.build_circuit()
    qaoa.ansatz = qaoa.ansatz.assign_parameters({qaoa.ansatz.parameters[i]: x[i] for i in range(p * 2)})
    qaoa.ansatz.measure_all()

    job = execute(qaoa.ansatz, Aer.get_backend('qasm_simulator'), shots=n_shots)
    result = job.result().get_counts()
    exp_value = compute_expectation_value(result, qaoa.max_cut)
    return exp_value


def callback(xk):
    global iters
    exp = check_result([i for i in xk])
    iters.append(exp)


random.seed(1234)

n_nodes = 10
p = 2
n_shots = 100
max_iter = 200

with open('max_cut_10.pkl', 'rb') as file:
    max_cut = pickle.load(file)

qaoa = QAOA(max_cut=max_cut, p=p, backend=Aer.get_backend('qasm_simulator'))

optimizers = ['L-BFGS-B', 'COBYLA', ]

for optimizer in optimizers:


iters = []

obj = qaoa.compute_minimum_eigenvalue(n_shots=n_shots)
init_point = random_init_parameters(p)
res_sample = minimize(obj, init_point, method='COBYLA', callback=callback, options={'maxiter': max_iter,
                                                                                      'disp': False})
print(iters)
print(res_sample)
print(check_result(check_result([i for i in res_sample['x']])))