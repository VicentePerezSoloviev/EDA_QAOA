from qiskit import QuantumCircuit, execute
from qiskit.circuit.library import QAOAAnsatz
from MaxCut import MaxCut


class QAOA:

    def __init__(self, max_cut: MaxCut, p: int, backend):
        assert p > 0, 'Number of layers must be greater that zero'

        self.ansatz = QuantumCircuit()
        self.max_cut = max_cut
        self.p = p
        self.backend = backend

    def build_circuit(self):
        cost_operator, _ = self.max_cut.get_operator()
        self.ansatz = QAOAAnsatz(cost_operator, reps=self.p)

    def compute_minimum_eigenvalue(self, n_shots):

        def f(theta):
            params = list(theta.values)
            self.build_circuit()
            self.ansatz = self.ansatz.assign_parameters({self.ansatz.parameters[i]: params[i] for i in range(self.p*2)})
            self.ansatz.measure_all()

            job = execute(self.ansatz, self.backend, shots=n_shots)
            result = job.result().get_counts()
            exp_value = compute_expectation_value(result, self.max_cut)

            return exp_value

        return f


def compute_expectation_value(counts, max_cut: MaxCut):
    avg = 0
    sum_count = 0
    for bitstring, count in counts.items():
        obj = max_cut.obj_fun(str(bitstring))
        avg += obj * count
        sum_count += count

    return avg / sum_count
