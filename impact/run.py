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
from timeit import default_timer as timer
import communicator

def model(x, p):
	return np.array(
		p[0]  * -(x - 1/1.0)**2 + 
		p[1]  *  (x - 1/2.0)**2 + 
		p[2]  * -(x - 1/3.0)**2 +
		p[3]  *  (x - 1/4.0)**3 +
		p[4]  * -(x - 1/5.0)**2 +
		p[5]  *  (x - 1/6.0)**2 +
		p[6]  * -(x - 1/7.0)**2 +
		p[7]  *  (x - 1/8.0)**3 +
		p[8]  * -(x - 1/9.0)**2 +
		p[9]  *  (x - 1/10.0)**2 +
		p[10] * -(x - 1/11.0)**2 +
		p[11] *  (x - 1/12.0)**3 +
		p[12] * -(x - 1/13.0)**2 +
		p[13] *  (x - 1/14.0)**2 +
		p[14] * -(x - 1/15.0)**2 
		, np.float64)

def obj_c(p):
	return computeRMSE(p)

def obj_py(p):
	ymodel = model(xdata, p)
	RMSE   = np.sqrt(np.mean((ymodel - ydata)**2))
	return RMSE

if '-h' in sys.argv or '--help' in sys.argv:
	print("Usage: %s <data_points> [launch_command[, ...]]\n"%(sys.argv[0]))
	print("\tdata_points          Number of data points to process.")
	print("\tmpi_commands         Any number of space separated arguments.")
	print("\t                     This will be run in the shell by the mpi")
	print("\t                     controller code. Should invoke whichever")
	print("\t                     scheduling script is necessary.\n")
	print("Example: %s 4096 sh pbs_script.sh"%(sys.argv[0]))

else:
	np.random.seed(1010)

	nProcesses, functions = communicator.init('./', 'controller.so', ' '.join(sys.argv[2:]))

	configureDataSize = functions['configureDataSize']
	sendData          = functions['sendData']
	computeRMSE       = functions['computeRMSE']
	finish            = functions['finish']

	actual_parameters = [
		1/1.0 ,  1/0.5, 1/-0.5,  1/15.0, 1/-15.0,
		1/8.0 , 1/-8.0,  1/9.5, 1/-4.5 , 1/-6.0 ,
		1/12.0, 1/-7.7,  1/7.7,  1/5.0 ,  1/1.0
	]

	guess_parameters = [
		4.0 ,  3.5, -1.5,  10.0, -10.0,
		4.0 , -2.0,  6.5, -1.5 , -4.0 ,
		2.0 , -3.7,  3.7,  7.0 ,  3.0
	]

	rng    = [-5.0, 5.0]
	points = int(sys.argv[1])

	

	xdata   = np.linspace(rng[0], rng[1], points)
	y       = model(xdata, actual_parameters)
	y_noise = 10 * np.random.normal(size=xdata.size)
	ydata   = y + y_noise

	configureDataSize(rng[0], rng[1], points)
	sendData(np.array(ydata, np.float64))

	x0 = guess_parameters

	start_c = timer()
	res_c = minimize(obj_c, x0, method='Nelder-Mead', tol=1e-6)
	end_c = timer()

	start_py = timer()
	res_py = minimize(obj_py, x0, method='Nelder-Mead', tol=1e-6)
	end_py = timer()

	# Shut down the MPI processes.
	finish()

	c_error      = np.sqrt(np.mean((ydata - model(xdata, res_c.x))**2))
	python_error = np.sqrt(np.mean((ydata - model(xdata, res_py.x))**2))

	print("Timing: ")
	print("\tC Error Minimization Took:      %fs"%(end_c - start_c))
	print("\tPython Error Minimization Took: %fs\n"%(end_py - start_py))

	print("Error: ")
	print("\tC:      %f"%(c_error))
	print("\tPython: %f"%(python_error))

