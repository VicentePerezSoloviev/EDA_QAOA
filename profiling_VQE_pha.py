import pandas as pd
import numpy as np
from qiskit.algorithms import QAOA, VQE
from continuous_qiskit_optimized import UMDAc as EDAc
from qiskit import Aer
import random
import time
import pickle
from qiskit.utils import QuantumInstance
from qiskit.circuit.library import TwoLocal
from qiskit.providers.aer.noise import depolarizing_error, NoiseModel, amplitude_damping_error, phase_damping_error

random.seed(1234)


with open('max_cut_10.pkl', 'rb') as file:
    max_cut = pickle.load(file)

qubit_op, _ = max_cut.get_operator()


def create_vector(num_params):
    vec = pd.DataFrame(columns=list(range(0, num_params)))
    vec['data'] = ['mu', 'std', 'min', 'max']
    vec = vec.set_index('data')
    vec.loc['mu'] = np.pi
    vec.loc['std'] = 0.5

    vec.loc['min'] = 0  # optional
    vec.loc['max'] = np.pi * 2  # optional
    return vec


ps = range(1, 7)
size_gens = [10, 20, 30]

filename = 'output_eda_elite_3_opt_vqe_pha.csv'
dt = pd.DataFrame(columns=['opt', 'it', 'p', 'size_gen', 'best_cost', 'time', 'gamma'])

gamma = 0.5
noise_model = NoiseModel()
error1 = depolarizing_error(gamma, 1)
error2 = depolarizing_error(gamma*2, 2)
noise_model.add_all_qubit_quantum_error(error1, ['rx', 'h', 'rz'])
noise_model.add_all_qubit_quantum_error(error2, ['cnot', 'cx'])

index = 0
for gamma in [10**-5, 10**-4, 10**-3, 10**-2, 10**-1, 0.9]:
    noise_model = NoiseModel()
    error = phase_damping_error(gamma)
    noise_model.add_all_qubit_quantum_error(error, ['rx', 'h', 'rz'])

    for p in ps:
        for size_gen in size_gens:
            for it in range(15):
                ansatz = TwoLocal(rotation_blocks='ry', entanglement_blocks='cz', reps=p)
                vector = create_vector(p*10 + 10)
                eda = EDAc(size_gen=size_gen, max_iter=100, dead_iter=20, alpha=0.7, vector=vector)

                counts = []
                values = []

                def store_intermediate_result(eval_count, parameters, mean, std):
                    counts.append(eval_count)
                    values.append(mean)


                vqe = VQE(ansatz, eda, callback=store_intermediate_result,
                          quantum_instance=QuantumInstance(backend=Aer.get_backend('statevector_simulator'),
                                                           noise_model=noise_model))
                # by default 1024 shots
                start_time = time.process_time()
                result = vqe.compute_minimum_eigenvalue(operator=qubit_op)  # result is VQEResult
                finish_time = time.process_time()

                dt.loc[index] = ['eda', it, p, size_gen, result.optimal_value, finish_time-start_time, gamma]
                print('eda', it, p, size_gen, result.optimal_value, finish_time-start_time, gamma)
                dt.to_csv(filename)
                index = index + 1
