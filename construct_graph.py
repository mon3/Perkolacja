# -*- coding: utf-8 -*-
import numpy as np
import pylab
import math
import matplotlib.pyplot as plt


def construct_graph(p, N):
    # N - number of all nodes
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

def clusters(adj_matrix, N):
    K = np.zeros(N) # vector klusterów


    changed = False
    value_for_change = 0
    index = 0

    for i in range(N):
        if (K[i] == 0):
            K[i] = i+1
        for j in range(i+1, N):
            if ((adj_matrix[i][j] == 1) & (K[j] == 0)):
                K[j] = K[i]
            elif ((adj_matrix[i][j] == 1) & (K[j] != 0) & (changed == False)):
                changed = True
                value_for_change = K[j]
                index = j
                old_value = K[i]
                K[i] = value_for_change
                for l in range(i+1, index):
                    if (K[l] == old_value):
                         K[l] = value_for_change
            elif ((adj_matrix[i][j] == 1) & (K[j] != 0) & (changed == True)):
                #value_for_change = K[j]
                K[j] = value_for_change
                
        changed = False
        value_for_change = 0 
        index = 0      
                
    return K


def nodes(K,N): # K- cluster, N - number of all nodes
    K_length = np.zeros(N)

    for i in range(N):
        K_length[i] = (K==(i+1)).sum()

    N_G = max(K_length)
    return N_G


def main():
    
    p = 0.3
    N = 1000
    # S - rozmiar klastra
    # S = N_G / N, gdzie N_G - liczba węzłów w największym klastrze
    # k - średni stopień, zależny od p
    # <k> = p(N-1)

    k_average = np.arange(0.0,6.0,0.1)

    S_array = []
    for i in k_average:
        probability = i/ (N-1)
        adj_matrix = construct_graph(probability,N)
        K = clusters(adj_matrix,N)
        N_G = nodes(K,N)
        S = N_G/N
        S_array.append(S)

   #print S_array

    plt.plot(k_average, S_array, 'o--',color='g', label='zaleznosc')
    plt.show()


    # adj_matrix = construct_graph(p,N)
    # print adj_matrix    

    # K = clusters(adj_matrix,N)
    # print K
    
    # N_G = nodes(K,N)
    # print N_G

if __name__ == "__main__":
    main()