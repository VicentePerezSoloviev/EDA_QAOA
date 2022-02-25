import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from qiskit.quantum_info import Pauli
from qiskit.opflow import PauliSumOp


class MaxCut:

    def __init__(self, n_nodes: int, adj_matrix=None):
        self.n_nodes = n_nodes
        self.adj_matrix = adj_matrix
        self.graph = nx.Graph()

    def random_graph(self, per_arcs: float):
        assert 0 < per_arcs < 1, 'Percentage of arcs must be a value between 0 and 1'

        nums = np.ones(self.n_nodes ** 2)
        nums[:int(len(nums)*per_arcs)] = 0
        np.random.shuffle(nums)
        nums = np.reshape(nums, (self.n_nodes, self.n_nodes))
        for i in range(self.n_nodes):
            for j in range(self.n_nodes):
                if i == j:
                    nums[i, j] = 0

        self.adj_matrix = nums

        self.graph = nx.from_numpy_matrix(self.adj_matrix)

    def plot_graph(self, filename=None):
        g = nx.from_numpy_matrix(self.adj_matrix)
        layout = nx.random_layout(g, seed=10)
        nx.draw(g, layout, node_color='blue')
        labels = nx.get_edge_attributes(g, 'weight')
        nx.draw_networkx_edge_labels(g, pos=layout, edge_labels=labels)

        if filename is not None:
            plt.savefig(filename)
        plt.show()

    def get_operator(self):
        """Generate Hamiltonian for the graph partitioning
        Notes:
            Goals:
                1 separate the vertices into two set of the same size
                2 make sure the number of edges between the two set is minimized.
            Hamiltonian:
                H = H_A + H_B
                H_A = sum\_{(i,j)\in E}{(1-ZiZj)/2}
                H_B = (sum_{i}{Zi})^2 = sum_{i}{Zi^2}+sum_{i!=j}{ZiZj}
                H_A is for achieving goal 2 and H_B is for achieving goal 1.
        Returns:
            PauliSumOp: operator for the Hamiltonian
            float: a constant shift for the obj function.
        """
        pauli_list = []
        shift = 0

        for i in range(self.n_nodes):
            for j in range(i):
                if self.adj_matrix[i, j] != 0:
                    x_p = np.zeros(self.n_nodes, dtype=bool)
                    z_p = np.zeros(self.n_nodes, dtype=bool)
                    z_p[i] = True
                    z_p[j] = True
                    pauli_list.append([-0.5, Pauli((z_p, x_p))])
                    shift += 0.5

        for i in range(self.n_nodes):
            for j in range(self.n_nodes):
                if i != j:
                    x_p = np.zeros(self.n_nodes, dtype=bool)
                    z_p = np.zeros(self.n_nodes, dtype=bool)
                    z_p[i] = True
                    z_p[j] = True
                    pauli_list.append([1, Pauli((z_p, x_p))])
                else:
                    shift += 1

        pauli_list = [(pauli[1].to_label(), pauli[0]) for pauli in pauli_list]
        return PauliSumOp.from_list(pauli_list), shift

    def obj_fun(self, string):
        obj = 0
        for i, j in self.graph.edges():
            if string[i] != string[j]:
                obj -= 1
        return obj
