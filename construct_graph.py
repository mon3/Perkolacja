# -*- coding: utf-8 -*-
import numpy as np
import pylab
import math
import matplotlib.pyplot as plt


def construct_graph(p, N):
    adjacency_matrix = np.zeros((N, N)) #tworzymy macierz sasiedztwa
    for i in range (N):
        for j in range (i+1, N):
            random_p = np.random.random()
            if (random_p < p):
                if(i==j):
                    adjacency_matrix[i][j] = 0
                else:
                    adjacency_matrix[i][j] = 1
                   # adjacency_matrix[j][i] = 1

    return adjacency_matrix


def main():
    
    p = 0.6
    N = 5

    adj_matrix = construct_graph(p,N)
    print adj_matrix    

if __name__ == "__main__":
    main()