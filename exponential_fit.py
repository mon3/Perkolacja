import numpy as np 
import math
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def func(x, a, b, c):
    return a * np.exp(-b * x) + c



if __name__ == "__main__": 

	k_av = [1.5,1.4,1.2]
	N = [100,300,1000]

	popt, pcov = curve_fit(func, N, k_av)
	print popt
	print pcov


	plt.figure()
	plt.axis([ 0,1200,1.1,1.6])
	ax = plt.gca()
	ax.set_autoscale_on(False)
	plt.grid()
	plt.plot(N, k_av, 'ko', label="symulacja")
	print func(N,*popt)
	plt.plot(N, func(N, *popt), 'r-', label="Dopasowana krzywa <k> = A*exp(-B*N)+C")
	plt.legend()
	
	plt.show()
