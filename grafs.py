# -*- coding: utf-8 -*-
import numpy as np
# import pylab
# import math
import matplotlib.pyplot as plt
# from numpy import genfromtxt
# import networkx as nx
# import igraph



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
	# g.vs['label'] = node_names  # or a.index/a.columns

	igraph.drawing.plot(g)

def construct_graph(p, N):
    # N - number of all nodes
    wynik = np.zeros((len(p), N, N))
    acceptance_probabilities = np.random.random((len(p), N, N))
    wynik[acceptance_probabilities < p[:, np.newaxis, np.newaxis]] = 1 
    #np.newaxis roszerza wektor o kolejny wymiar wedlug" : "
    for m in range(len(p)):
        wynik[m, :, :] = np.triu(wynik[m, :, :], 1) 
    return wynik


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
                value_for_change = K[i] # K[i] zamiast K[j]
                index = j
                old_value = K[j] #zamiast K[i] daje K[j]
                K[j] = value_for_change #zamiast K[i] daję K[j]
                for l in range(N): #zmieniam granice -> musi sprawdzic wszystkie wezly
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
    """
    calculates number of nodes in largest cluster
    [1,2,1,1,3,4,1]
    {1: 4, 2: 1, 3: 1, 4: 1}.max()
    """
    slownik = {k: 0.0 for k in K}
    for k in K:
        slownik[k] += 1

    N_G = max(slownik.values())
    return N_G



if __name__ == "__main__": 
    p = 0.6
    N = 100
    # S - rozmiar klastra
    # S = N_G / N, gdzie N_G - liczba węzłów w największym klastrze
    # k - średni stopień, zależny od p
    # <k> = p(N-1)


    k_average = np.arange(0.0,6.0,0.1)
   

    

    n = liczba_iteracji = 100 #do liczenia sredniej
    
   
    S = np.zeros((len(k_average), liczba_iteracji))

    for j in range(liczba_iteracji):
        probability = k_average/ (N-1)
        adj_matrix = construct_graph(probability,N)
            #print adj_matrix
        for m in range(len(probability)):
            K = clusters(adj_matrix[m,:,:],N)
            N_G = nodes(K,N)
            S[m, j] = N_G/N
                #print S
    print(S.shape)
    print(S)

    # S_average - srednie, S_stdev - odchylenia standardowe
    S_average = np.sum(S, axis=1)/n   
    S_average_square = np.sum(S**2, axis=1)/n 
    S_stdev = np.sqrt(S_average_square - S_average**2)
    
    plt.plot(k_average, S_average, 'o',color='g', label='zaleznosc')
    plt.errorbar(k_average, S_average,  yerr=S_stdev)

    plt.show()



