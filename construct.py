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
    # nx.draw(gr) # edited to include labels
    nx.draw_networkx(gr)
    # now if you decide you don't want labels because your graph
    # is too busy just do: nx.draw_networkx(G,with_labels=False)
    plt.show() 

def show_graph(A):
	# Create graph, A.astype(bool).tolist() or (A / A).tolist() can also be used.
	g = igraph.Graph.Adjacency((A > 0).tolist())
	print g
	# Add edge weights and node labels.
	g.es['weight'] = A[A.nonzero()]
	# g.vs['label'] = node_names  # or a.index/a.columns

	igraph.drawing.plot(g)

def construct_graph(p, N):
    # N - number of all nodes
    adjacency_matrix = np.zeros((N, N)) #tworzymy macierz sasiedztwa
    #sum = 0
    for i in range (N):
        for j in range (i+1, N):
            random_p = np.random.random()
            if (random_p < p):
                if(i==j):
                    adjacency_matrix[i][j] = 0
                else:
                    adjacency_matrix[i][j] = 1
                   # sum +=1
                   # adjacency_matrix[j][i] = 1

    #print "suma 1: ",sum
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
    K_length = np.zeros(N)

    for i in range(N):
        K_length[i] = (K==(i+1)).sum()

    N_G = max(K_length)
    return N_G


# def main():
if __name__ == "__main__": 
    p = 0.6
    N = 1000
    # S - rozmiar klastra
    # S = N_G / N, gdzie N_G - liczba węzłów w największym klastrze
    # k - średni stopień, zależny od p
    # <k> = p(N-1)


    k_average = np.arange(0.0,6.0,0.05)
   # S_error = []

    


    S_array = []
    for i in k_average:
        probability = i/ (N-1)
        adj_matrix = construct_graph(probability,N)
        #print adj_matrix
        K = clusters(adj_matrix,N)
        N_G = nodes(K,N)
        S = N_G/N
        #print S
        S_array.append(S)
       # S_error.append(np.sqrt(S))

   	#print S_array
    
    plt.plot(k_average, S_array, 'o',color='g', label='zaleznosc')
   # plt.errorbar(k_average, S_array,  yerr=S_error)

    plt.show()

    # k = 3.
    # probability = k/ (N-1)
    # adj_matrix = construct_graph(probability,N)
    # print adj_matrix  


    # sprawdzenie dla graf 4 węzłów {1,2,3,4}, w którym są
	# trzy połączenia: 1-3, 3-4, 4-2.
    # adj_matrix = np.zeros((N, N))
    # adj_matrix[0][2] = 1
    # adj_matrix[1][3] = 1
    # adj_matrix[2][3] = 1

    # print adj_matrix



    # adj_matrix = np.zeros((N, N))
    # adj_matrix[0][1] = 1
    # adj_matrix[1][9] = 1
    # adj_matrix[1][6] = 1
    # adj_matrix[1][8] = 1
    # adj_matrix[4][6] = 1
    # adj_matrix[3][5] = 1
    # adj_matrix[6][8] = 1
    




    # print adj_matrix


    K = clusters(adj_matrix,N)
    print K
    
    N_G = nodes(K,N)
    print N_G
    #plt.imshow(adj_matrix, interpolation='nearest')
   # plt.colorbar()
    #plt.show()


   # show_graph(adj_matrix)



  #  main()[]