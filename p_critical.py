# -*- coding: utf-8 -*-
import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def prosta(x,a,b):
    return -a*x+b

def clusters(K,N): # K- cluster, N - number of all nodes
    K_length = np.zeros(N)

    for i in range(N):
        K_length[i] = (K==(i+1)).sum()


    K_length.sort()
    K_length = K_length[::-1]

    return K_length

if __name__ == "__main__":

    k_limit = np.arange(0.3,1.6,0.1)

    listy_S = {}
    listy_k = {}
    listy_p = {}
    listy_k_crit = {}

    # f istnieje tylko wewnatrz tego scope, a pozniej robi 'close'
    with h5py.File("grafy_data.hdf5") as f:
        for data_name in f:
            dataset = f[data_name]
            i = dataset.attrs['i']
            N = dataset.attrs['N']
            s = dataset.attrs['S_avg']
            if N in listy_S.keys():
                listy_S[N].append(s)
                listy_k[N].append(i)
                listy_k_crit[N].append(i-1.2)
                listy_p[N].append((i-1.2)/999.)
            else:
                listy_S[N] = [s]
                listy_k[N] = [i]
                listy_p[N] = [i/999.]
                listy_k_crit[N]= [i-1.2]

                # listy_k[N].append(i)
                # lista_S.append(S_avg)
                # pierwszy_cluster = dataset[0]
                # print i, N, pierwszy_cluster
                # print(dataset)
                # for attribute in dataset.attrs:
                #     print(attribute, dataset.attrs[attribute])

        # for N in 1000:
        #     if N > 70:
        #         print(N)
               # plt.plot(listy_k[N], listy_S[N], "o--", label=N)
        plt.plot(np.log(np.abs(listy_p[1000])),np.log(listy_S[1000]))
                # S_max_arg = np.argmax(np.array(listy_S[N]))
                # S_max = listy_S[N][S_max_arg]
                # k_max= listy_k[N][S_max_arg]
               # plt.plot(k_max, S_max, "o", lw=5) # punkty max
               # plt.text(k_max, S_max, str(k_max)) # wspolrzedne punktow max
                #plt.legend()
        plt.show()


       #  lista_y = []
       #  p_crit = 1.2/999
       #  for i in listy_p[1000]:
       #      lista_y.append(i-p_crit)

       # # print lista_y
       #  # lista_p = []
       #  # for k_arg in listy_k[1000]:
       #  #     lista_p.append(np.abs((k_arg-1.2)/999.))

       #  lista_y = lista_y[lista_y!=0]
       #  listy_S[1000] = listy_S[1000][lista_y!=0]
       #  x_dane = np.log(lista_y)
       #  y_dane = np.log(listy_S[1000])




       #  plt.plot(y_dane, x_dane, "o--")
       #  plt.show()
       