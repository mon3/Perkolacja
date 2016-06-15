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
    
    index = 0

    for i in range(N):
        if (K[i] == 0):
            K[i] = i+1
            value_for_change = 0
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
            	#sprawdzic, czy tutaj wchodzi kiedykolwiek
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

   # print "K_length: ", K_length
    N_G = max(K_length)
    return N_G


def second_node(K,N): # K- cluster, N - number of all nodes
    K_length = np.zeros(N)

    for i in range(N):
        K_length[i] = (K==(i+1)).sum()

   
    K_length.sort()
    K_length = K_length[::-1]

    N_g= K_length[1] #eliminujemy najdluzszy kluster
    return N_g


# def main():
if __name__ == "__main__": 
    p = 0.6
    N = 10
    # S - rozmiar klastra
    # S = N_G / N, gdzie N_G - liczba węzłów w największym klastrze
    # k - średni stopień, zależny od p
    # <k> = p(N-1)


    k_average = np.arange(0.0,6.0,0.1)
   # S_error = []

    
   # k_limit = np.arange(0.9,1.1,0.002)
    k_limit = np.arange(0.3,1.6,0.1)



    n = 100
    
   
    

    # for N in (100,300,500,1000): # pętla po x: 
    #     S_average = []
    #     S_stdev = []
    #     for i in k_limit: # pętla po x: 
    #       #  print i
    #         S_array = []
    #         average = 0
    #         average_square = 0 
    #         for j in range(n):
    #             probability = i/ (N-1)
    #             adj_matrix = construct_graph(probability,N)
    #                 #print adj_matrix
    #             K = clusters(adj_matrix,N)
    #            # print "K: ", K
    #             #N_G = nodes(K,N)
    #             N_G = second_node(K,N)
    #             S = N_G/N
    #                 #print S
    #             S_array.append(S)
    #            # S_error.append(np.sqrt(S))
    #             average += S #sum(S_array)
    #             average_square += S**2
    #         S_average.append(average/n)
    #         S_stdev.append(np.sqrt(average_square / n - (average / n)**2))
        
    #     # import ipdb; ipdb.set_trace();
    #     # print len(k_limit, len(S_average)
    #     plt.plot(k_limit, S_average, 'o--',  label=("$N=%d$"%N))
    #     # S_stdev.append(np.sqrt(S**2 - average**2))
   	# #print S_array
    
    # # plt.plot(k_average, S_average, 'o',color='g', label='zaleznosc')
    # # plt.errorbar(k_average, S_average,  yerr=S_stdev)

    

    # #plt.plot(k_limit, S_average, 'o',color='g', label='zaleznosc')
    # #plt.errorbar(k_limit, S_average,  yerr=S_stdev)
    # #plt.errorbar(k_limit, S_average)
    # plt.title(u"Zależność <s>(<k>)")
    # plt.xlabel('<k>')
    # plt.ylabel('<s>')
    # plt.grid()
    # plt.legend(loc='best')
    # plt.savefig("sredni_klaster.png")
    # plt.savefig("sredni_klaster.pdf")
    # plt.show()

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


    # sprawdzenie na konkretnej macierzy - dziala poprawnie
    adj_matrix = np.zeros((N, N))
    # adj_matrix[0][1] = 1
    # adj_matrix[1][9] = 1
    # adj_matrix[1][6] = 1
    # adj_matrix[1][8] = 1
    # adj_matrix[4][6] = 1
    # adj_matrix[3][5] = 1
    # adj_matrix[6][8] = 1

    adj_matrix[0][9] = 1
    adj_matrix[1][3] = 1
    adj_matrix[1][6] = 1
    adj_matrix[1][8] = 1
    adj_matrix[4][6] = 1
    adj_matrix[3][5] = 1
    adj_matrix[6][8] = 1
    




    print "macierz sąsiedztwa " , adj_matrix


    K = clusters(adj_matrix,N)

    print "wektor klastrów ",K

    K_length = np.zeros(N)
    for i in range(N):
        K_length[i] = (K==(i+1)).sum()

    print "tablica wskazująca ile jest węzłów dla klastrów o numerach równych indeksom tablicy powiększonym o 1 ",K_length

    
    N_G = nodes(K,N)

    print "Liczba węzłów odpowiadająca maksymalnemu klastrowi: ",N_G
    
    plt.show()





