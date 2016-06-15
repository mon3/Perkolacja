# -*- coding: utf-8 -*-
import numpy as np
import pylab
import math
import matplotlib.pyplot as plt
from numpy import genfromtxt
import networkx as nx
import igraph



def show_graph(adjacency_matrix):
    # given an adjacency matrix use networkx and matlpotlib to plot the graph

    rows, cols = np.where(adjacency_matrix == 1)
    edges = zip(rows.tolist(), cols.tolist())
    gr = nx.Graph()
    gr.add_edges_from(edges)
    nx.draw_networkx(gr)
   
    plt.show() 

def show_graph(A):
	
	g = igraph.Graph.Adjacency((A > 0).tolist())
	print g
	# Add edge weights and node labels.
	g.es['weight'] = A[A.nonzero()]
	

	igraph.drawing.plot(g)

def construct_graph(p, N):
    # N - number of all nodes
    adjacency_matrix = np.zeros((N, N)) #tworzymy macierz sasiedztwa
    sum = 0
    for i in range (N):
        for j in range (i+1, N):
            random_p = np.random.random()
            if (random_p < p):
                if(i==j):
                    adjacency_matrix[i][j] = 0
                else:
                    adjacency_matrix[i][j] = 1
                    sum +=1
                   # adjacency_matrix[j][i] = 1

    print "suma 1: ",sum
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


if __name__ == "__main__": 
    p = 0.6
    N = 300
    # S - rozmiar klastra
    # S = N_G / N, gdzie N_G - liczba węzłów w największym klastrze
    # k - średni stopień, zależny od p
    # <k> = p(N-1)

    k_average = np.arange(0.0,6.0,0.05)

    S_array = []
    for i in k_average:
        probability = i/ (N-1)
        adj_matrix = construct_graph(probability,N)
        K = clusters(adj_matrix,N)
        N_G = nodes(K,N)
        S = N_G/N
        S_array.append(S)

   
    
    plt.plot(k_average, S_array, 'o--',color='g', label='zaleznosc')
    plt.show()

  