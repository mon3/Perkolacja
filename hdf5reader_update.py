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


    #tutaj byl blad - ponizsze linijki już są nieptorzebne
    # K_length.sort()
    # K_length = K_length[::-1]

    return K_length

if __name__ == "__main__":

    k_limit = np.arange(0.3,1.6,0.1)

    listy_S = {}
    listy_k = {}
    listy_p = {}
    listy_k_crit = {}
    # f istnieje tylko wewnatrz tego scope, a pozniej robi 'close'
    with h5py.File("target.hdf5") as f:
        for data_name in f:
            dataset = f[data_name]
            i = dataset.attrs['i']
            N = dataset.attrs['N']
            s = dataset.attrs['S_avg']
            if N in listy_S.keys():
                listy_S[N].append(s)
                listy_k[N].append(i)
            else:
                listy_S[N] = [s]
                listy_k[N] = [i]

                # listy_k[N].append(i)
                # lista_S.append(S_avg)
                # pierwszy_cluster = dataset[0]
                # print i, N, pierwszy_cluster
                # print(dataset)
                # for attribute in dataset.attrs:
                #     print(attribute, dataset.attrs[attribute])



        for N in listy_S.keys():
            if N > 70:
                print(N)
                plt.plot(listy_k[N], listy_S[N], "o--", label=N)
                S_max_arg = np.argmax(np.array(listy_S[N]))
                S_max = listy_S[N][S_max_arg]
                k_max= listy_k[N][S_max_arg]
                plt.plot(k_max, S_max, "o", lw=5)
                plt.text(k_max, S_max, str(k_max))
        plt.legend()
        plt.show()

        

        if 1000 in listy_S.keys():
            n = 100
            N = 1000
            dane = f['N1000_k1.200000']
            #print  dane[1]
            number_clusters_per_size_array = np.zeros((n,N), dtype = int)
           # K_length = np.zeros((n,N))
            klasterki = np.zeros((n,N))
            K =np.zeros((n,N))
            for i in range(n):
                K[i]= dane[i]
     
            K_append = np.zeros((n,N))
            for i in range(n):

                if (i == 1):   
                    print "dane dla pojedyńczego klastra podanego w postaci(indeks tablicy i - nr węzła, K[i] -numer klastra, do którego należy ten węzeł "
                    print K[i]

                K_length = np.zeros(N)
                for j in range(N):
                    K_length[j] = (K[i] ==(j+1)).sum()
               # K_length = K_length[::-1]
                K_append[i] = K_length

               
                if (i == 1):   
                    print "dane dla pojedyńczego wektora klastrów po zliczeniu liczby węzłów dla klastrów wchodzących w skład tego wektora"
                    print "K_length[j] odpowiada liczbie węzłów, które wchodzą do klastra o przypisanym jemu numerze(oznaczeniu), równym (j+1)" 
                    print K_append[i]  
                    #print "suma = ", np.sum(K_append[i])


                #print "K_append[0]",K_append[0]

            histogram = np.zeros(N)
            histograms = np.zeros((n,N))


            # zliczam histogramy dla poszczegoólnych wektorów klastrów
            for i in range(n):
                for j in range(N):
                    value=K_append[i][j]
                    histograms[i][value] += 1


            # zliczam zsumowane histogramy
            for i in range(n):
                for j in range(N):
                    histogram[j]+= histograms[i][j]

            print "histogram wspolny: ", histogram

            # #normalizacja histogramu wspólnego
            # for i in range(N):
            #     histogram[i] = histogram[i]/(i*n)

            #print "histogram znormalizowany: ", histogram

            # for i in range(N):
            #     value = K_append[0][i]
            #     histogram[value]+=1

            # histogram wspólny dla wektoru klastrów

            histogram = histogram[histogram>0]
            #x_data = np.arange(N) +1
           
            y_data = histogram
            x_data = 1 + np.arange(len(histogram))
            print "y_data ", y_data

           
            x = np.log(x_data)
            y = np.log(y_data)

            print "x = ", x
            print "y = ", y

            # ind - daję warunek, żeby x były z konkretnego przedziałuu w celu eliminacji inf oraz 0
            ind = (x>0.9) & (x < 3.8)
            x_fit = x[ind]
            y_fit = y[ind]

            plt.plot(x,y,'bo',label='symulacja')
        

            
            # fituję prostą i wypisuję współczynnik nachylenia
            params, pcov = curve_fit(prosta, x_fit, y_fit)
            print "parametr", params[0]

            Y_fit = []
            for k in x_fit:
                Y_fit.append(-params[0]*k+params[1])


            # dane do dopasowanej prostej    
            plt.plot(x_fit,Y_fit,'g-',label='fit')
            plt.legend()
            plt.grid()
            plt.title(u"Zależność ln(N(s))")
            plt.xlabel('lnS')
            plt.ylabel('ln(N(S))')
            plt.show()