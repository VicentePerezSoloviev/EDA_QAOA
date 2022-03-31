#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import math
from itertools import combinations
from scores import Scores


class QUBO:
    """
    QUBO formulation for the Bayesian Network Structure Learning problem, solved with Fujitsu Digital Annealer.
    """
    # Note that scores are in positive values, so the minimum value found is the optimum. Moreover, the
    # penalization are positive values, which will increase the result in case they are met.

    def __init__(self, scores_class, m=2):
        """
        Initialization of the QUBO formulation for the Bayesian Network Structure Learning problem.
        :param scores_class: scores class.
        :param m: integer. Maximum in-degree. By default 2.
        """

        assert isinstance(scores_class, Scores), 'Weights must be a dictionary with tuples as keys and float as values.'

        self.weights = scores_class.scores  # (target, parent, parent) = float  -> target | parent, parent
        self.identity = scores_class.identity
        n = len(self.identity)

        assert isinstance(m, int), 'Maximum in-degree must be of type integer.'
        assert isinstance(n, int), 'Number of nodes must be of type integer.'

        self.size = int(n*(n-1) + (n*(n-1))/2 + n*math.ceil(math.log(m+1, 2)))
        self.n = n
        self.m = m
        self.mu = math.ceil(math.log(m+1, 2))

        self.Q = np.zeros((self.size, self.size))
        self.rest = 0

    def q(self, trans_penalization, consist_penalization, max_penalization):
        """
        Q matrix building
        :param trans_penalization: Penalization for all the H_trans hamiltonian term.
        :param consist_penalization: Penalization for all the H_consist hamiltonian term.
        :param max_penalization: Penalization for the H_max hamiltonian term.
        :return: Q matrix update
        """

        self.h_score()
        self.h_cycle(trans_penalization, consist_penalization)
        self.h_max(max_penalization)

    def set(self, row, col, value):
        """
        Sets a value in a concrete position of the Q matrix. It sums a value to the actual value of matrix.
        :param row: Row in Q matrix.
        :param col: Col in Q matrix.
        :param value: Value to be summed.
        :return: Q matrix update
        """

        aux = self.Q[int(row), int(col)]

        assert self.Q[int(row), int(col)] == self.Q[int(col), int(row)], 'ERROR setting values in Q'

        self.Q[int(row), int(col)] = aux + value
        if row != col:  # for diagonal do not replicate
            self.Q[int(col), int(row)] = aux + value  # symmetric Q matrix

    def h_score(self):
        """
        H_score hamiltonian building
        :return: Q matrix update
        """
        # modification: take into account w_i({null})

        for i in range(self.n):
            array = [w for w in range(self.n) if w != i]

            # 2 parents
            for comb in combinations(array, 2):
                score = self.weights[i, comb[0], comb[1]] + self.weights[i] - \
                        self.weights[i, comb[0]] - self.weights[i, comb[1]]  # w_i({null})
                self.set(self.map_adj_vec(comb[0], i), self.map_adj_vec(comb[1], i), score)

            # 1 parent
            for w in array:
                score = self.weights[i, w] - self.weights[i]  # subtract w_i({null})
                self.set(self.map_adj_vec(w, i), self.map_adj_vec(w, i), score)

            # no parent
            self.rest = self.rest + self.weights[i]  # if no parents then score is w_i({null})

    def h_cycle(self, penalization_trans, penalization_consist):
        """
        H_cycle hamiltonian building.
        :param penalization_trans: Penalization term for the H_trans hamiltonian.
        :param penalization_consist: Penalization term for the H_consist hamiltonian.
        :return: Q matrix update.
        """

        self.h_trans(penalization_trans)
        self.h_consist(penalization_consist)

    def h_trans(self, penalization):
        """
        H_trans hamiltonian building.
        :param penalization: Penalization term
        :return: Q matrix update.
        """

        for i in range(self.n):
            for j in range(i+1, self.n):
                for k in range(j+1, self.n):
                    # rik + rij*rjk - rij*rik - rjk*rik
                    self.set(self.map_r_vec(i, k), self.map_r_vec(i, k), penalization)
                    self.set(self.map_r_vec(i, j), self.map_r_vec(j, k), penalization)
                    self.set(self.map_r_vec(i, j), self.map_r_vec(i, k), -penalization)
                    self.set(self.map_r_vec(j, k), self.map_r_vec(i, k), -penalization)

    def h_consist(self, penalization):
        """
        H_consist hamiltonian building.
        :param penalization: Penalization term
        :return: Q matrix update.
        """

        for i in range(self.n):
            for j in range(i + 1, self.n):
                # dji*rij + dij - dij*rij
                self.set(self.map_adj_vec(j, i), self.map_r_vec(i, j), penalization)
                self.set(self.map_adj_vec(i, j), self.map_adj_vec(i, j), penalization)
                self.set(self.map_adj_vec(i, j), self.map_r_vec(i, j), -penalization)

    def h_max(self, penalization):
        """
        H_max hamiltonian building.
        :param penalization: Penalization term
        :return: Q matrix update.
        """

        for i in range(self.n):
            # ( 2 - (sum(dji)) - (sum( 2**(l-1) * yil )) ) ** 2
            array = [[self.m]]
            # adj matrix
            d = [self.map_adj_vec(j, i) for j in range(self.n) if i != j]  # col of in-degrees
            for j in d:
                array.append([j, -1])

            # y matrix
            for j in range(1, self.mu+1):
                y_ = pow(2, j - 1) * (-1)
                array.append([self.map_y_vec(i, j-1), y_])

            # once the indexes are calculated, apply to Q matrix
            for j in range(len(array)):
                for k in range(len(array)):
                    # add penalization
                    from_ = array[j]
                    to_ = array[k]

                    if len(from_) == 2 and len(to_) == 2:
                        self.set(from_[0], to_[0], from_[1]*to_[1]*penalization)
                    elif len(from_) == 1 and len(to_) == 2:
                        self.set(to_[0], to_[0], from_[0]*to_[1]*penalization)
                    elif len(from_) == 2 and len(to_) == 1:
                        self.set(from_[0], from_[0], from_[1]*to_[0]*penalization)
                    else:
                        # case of len=1 * len=1
                        self.rest = self.rest + from_[0]*to_[0]*penalization

    def map_adj_vec(self, row, col):
        """
        Return the qubit associated to the row and col of the adj matrix.
        :param row: Row in the adj matrix
        :param col: Col in the adj matrix
        :return: Qubit associated in vector of qubit
        """

        assert row != col, "Diagonal adjacency indexes must not be taken into account"

        if col > row:
            return (row * self.n) + col - (row + 1)
        else:
            return (row * self.n) + col - row

    def map_vec_adj(self, index):
        """
        Returns the roz and col associated to the index in the vector
        :param index: index in qubit vector
        :return: tuple with row and col
        """

        assert index < self.n * (self.n - 1), "Index should be lower than the total quantity of qubits for adj matrix"

        index_new = 0
        for i in range(self.n):
            for j in range(self.n):
                if i != j:
                    if index == index_new:
                        return i, j
                    else:
                        index_new = index_new + 1

    def map_r_vec(self, row, col):
        """
        Return the qubit associated to the row and col of the r matrix.
        :param row: row in the r matrix.
        :param col: col in the r matrix.
        :return: Qubit associated in vector of qubit.
        """

        assert row < col, "r matrix is upper triangular"

        aux = self.n * (self.n - 1)
        return aux + (row * self.n) + col - int(((row + 2) * (row + 1)) / 2)

    def map_y_vec(self, row, col):
        """
        Return the qubit associated to the row and col of the y matrix.
        :param row: row in the y matrix.
        :param col: col in the y matrix.
        :return: Qubit associated in vector of qubit.
        """

        assert col < self.m, "number of columns in y matrix must be equal or lower than maximum in-degree"

        aux = self.n * (self.n - 1) + ((self.n * (self.n - 1))/2)
        return aux + (row * self.m) + col

    def export_matrix(self, path, delimiter):
        """
        Show the BN structure dynamically.
        :param delimiter: String delimiter to export the matrix
        :param path: name of the structure
        :return: open browser to show html
        """

        np.savetxt(path, self.Q, fmt='%.5f', delimiter=delimiter)

    def f(self, vector):
        assert len(vector) == self.size, 'Length of solution must math size of problem. Size = ' + str(self.size)

        transpose = np.transpose(vector)
        aux = np.matmul(vector, self.Q)
        return np.matmul(aux, transpose)

    def check_symmetric(self):
        return (self.Q.transpose() == self.Q).all()
