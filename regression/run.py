import numpy.ctypeslib as ctl
import numpy as np
import ctypes
from ctypes import *
import code
import scipy.optimize
import sys
from scipy.optimize import curve_fit
from scipy.optimize import minimize 
from random import randint as rand
from random import uniform as randf
from matplotlib import pyplot as plt
from timeit import default_timer as timer
import communicator

np.random.seed(1010)

nProcesses, functions = communicator.init('./', 'testlib.so', ' '.join(sys.argv[2:]))

configureDataSize = functions['configureDataSize']
sendData          = functions['sendData']
computeRMSE       = functions['computeRMSE']
finish            = functions['finish']

actual_parameters = [
	1.0 ,  0.5, -0.5,  15.0, -15.0,
	8.0 , -8.0,  9.5, -4.5 , -6.0 ,
	12.0, -7.7,  7.7,  5.0 ,  1.0
]

guess_parameters = [
	4.0 ,  3.5, -1.5,  10.0, -10.0,
	4.0 , -2.0,  6.5, -1.5 , -4.0 ,
	2.0 , -3.7,  3.7,  7.0 ,  3.0
]

rng    = [-4.0, 4.0]
points = int(sys.argv[1])

def model(x, p):
	return np.array(
		p[0] * x**2 + p[1] * x**2.5 + p[2] * x**3 + p[3] ** x**3.5 + p[4] * x**4 +
		p[5] * x**(-2) + p[6] * x**(-2.5) + p[7] * x**(-3) + p[8] ** x**(-3.5) + p[9] * x**(-4) +
		p[10] * np.exp(-x) + p[11] * np.exp(-2*x) + p[12] * np.exp(-3*x) + p[13] * np.exp(-4*x) +
		p[14] * np.exp(-5*x)
		, np.float64)

def obj_c(p):
	ymodel = model(xdata, p)
	RMSE   = computeRMSE(ymodel)
	return RMSE

def obj_py(p):
	ymodel = model(xdata, p)
	RMSE   = np.sqrt(np.mean((ymodel - ydata)**2))
	return RMSE

xdata   = np.linspace(rng[0], rng[1], points)
y       = model(xdata, actual_parameters)
y_noise = 10 * np.random.normal(size=xdata.size)
ydata   = y + y_noise

configureDataSize(rng[0], rng[1], points)
sendData(np.array(ydata, np.float64))

x0  = guess_parameters

start_c = timer()
res = minimize(obj_c, x0, method='Nelder-Mead', tol=1e-6)
end_c = timer()

start_py = timer()
res = minimize(obj_py, x0, method='Nelder-Mead', tol=1e-6)
end_py = timer()

# Shut down the MPI processes.
finish()

print("Timing: ")
print("\tC Error Minimization Took:      %fs"%(end_c - start_c))
print("\tPython Error Minimization Took: %fs"%(end_py - start_py))

yfit = model(xdata, res.x)
plt.plot(xdata, ydata, 'o', xdata, yfit, '-')
plt.show()

