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

def model(x, p):
	return np.array(p[0] * np.exp(p[1] * x) + p[2], np.float64)

def obj_c(p):
	ymodel = model(xdata, p)
	RMSE   = computeRMSE(ymodel)
	return RMSE

def obj_py(p):
	ymodel = model(xdata, p)
	RMSE   = np.sqrt(np.mean((ymodel - ydata)**2))
	return RMSE

xdata   = np.linspace(-4, 4, int(sys.argv[1]))
y       = model(xdata, [5, 1, 4.0])
y_noise = 10 * np.random.normal(size=xdata.size)
ydata   = y + y_noise

configureDataSize(len(xdata))
sendData(np.array(ydata, np.float64))

x0  = [2, 0, 1]

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

