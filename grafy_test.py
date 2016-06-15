# -*- coding: utf-8 -*-
import numpy as np
import grafy_przypadkowe as gp



def test_N_values():
    N = 4
    adj_matrix = np.zeros((N, N))
    adj_matrix[0][2] = 1
    adj_matrix[1][3] = 1
    adj_matrix[2][3] = 1


    K = gp.clusters(adj_matrix, 4)
    assert gp.nodes(K,4) == 4

class TestAdjacencyMatrix:
    def test_upper_triangular(self):
        k = np.array([3.0])
        N = 100
        probability = k/ (N-1)
        adj_matrix = gp.construct_graph(probability,N)
        print adj_matrix.shape
        czy_dane_wejscie_sie_zgadza = adj_matrix == np.triu(adj_matrix,1)
        print(czy_dane_wejscie_sie_zgadza)
        assert czy_dane_wejscie_sie_zgadza.all() 

    def test_matrix_values(self):
        k = np.array([3.0])
        N = 100
        probability = k/ (N-1)
        adj_matrix = gp.construct_graph(probability,N)
        average = np.sum(adj_matrix)*2/ (N*(N-1))

        print probability
        assert np.isclose(average, probability[0], rtol = 0.1)


# sprawdzenie na konkretnej macierzy - dziala poprawnie
# adj_matrix = np.zeros((N, N))
# adj_matrix[0][1] = 1
# adj_matrix[1][9] = 1
# adj_matrix[1][6] = 1
# adj_matrix[1][8] = 1
# adj_matrix[4][6] = 1
# adj_matrix[3][5] = 1
# adj_matrix[6][8] = 1
