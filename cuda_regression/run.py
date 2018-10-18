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
from mpl_toolkits.mplot3d import Axes3D
import math
import comm

np.random.seed(1010)

functions = comm.init('./', 'regression.so')

configure = functions['configure']
finish    = functions['finish']
getRMSE   = functions['getRMSE']

actual_parameters = np.array([
	1/1.0 ,  1/0.5, 1/-0.5,  1/15.0, 1/-15.0,
	1/8.0 , 1/-8.0,  1/9.5, 1/-4.5 , 1/-6.0 ,
	1/12.0, 1/-7.7,  1/7.7,  1/5.0 ,  1/1.0
], np.float32)

guess_parameters = np.array([
	4.0 ,  3.5, -1.5,  10.0, -10.0,
	4.0 , -2.0,  6.5, -1.5 , -4.0 ,
	2.0 , -3.7,  3.7,  7.0 ,  3.0
], np.float32)

rng     = [-5.0, 5.0]
points  = eval(sys.argv[1])
blocks  = eval(sys.argv[2])
threads = eval(sys.argv[2])

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
		, np.float32)


def obj_c(p):
	return getRMSE(np.array(p, np.float32))

def obj_py(p):
	ymodel = model(xdata, p)
	RMSE   = np.sqrt(np.mean((ymodel - ydata)**2))
	return RMSE


xdata   = np.array(np.linspace(rng[0], rng[1], points), np.float32)
y       = model(xdata, actual_parameters)
y_noise = 10 * np.random.normal(size=xdata.size)
ydata   = np.array(y + y_noise, np.float32)

configure(rng[0], rng[1], points, ydata, threads, blocks)

x0 = guess_parameters

print("Beginning Minimization\n")

start_c = timer()
res_c   = minimize(obj_c, x0, method='Nelder-Mead', tol=1e-6)
end_c   = timer()

start_py = timer()
res_py   = minimize(obj_py, x0, method='Nelder-Mead', tol=1e-6)
end_py   = timer()

finish()

cuda_error   = np.sqrt(np.mean((ydata - model(xdata, res_c.x))**2))
python_error = np.sqrt(np.mean((ydata - model(xdata, res_py.x))**2))

print("Done\n")

print("Results:")
print("\tCUDA:")
print("\t\tTime: %fs"%(end_c - start_c))
print("\t\tRMSE: %f\n"%(cuda_error))
print("\tPython:")
print("\t\tTime: %fs"%(end_py - start_py))
print("\t\tRMSE: %f\n"%(python_error))

print("CUDA was %f times faster than Python"%((end_py - start_py) / (end_c - start_c)))
# yfit_c  = model(xdata, res_c.x)
# yfit_py = model(xdata, res_py.x)
# plt.plot(xdata, ydata, 'o', xdata, yfit_c, '-', xdata, yfit_py, '-')
# plt.show()










# c_timings = []
# c_threads = []
# c_blocks  = []

# thread_count = 32
# while thread_count <= 1024:
# 	block_count = 32
# 	while block_count <= 1024:
# 		points = 1024*1024
# 		ppt = float(points) / float(thread_count * block_count)
# 		while math.floor(ppt) != ppt:
# 			points -= 1
# 			ppt = float(points) / float(thread_count * block_count)

		

		
		
# 		finish()
# 		c_threads.append(thread_count)
# 		c_blocks.append(block_count)
# 		c_timings.append((1.0 / (end_c - start_c)) * (points / (1024*1024)))

# 		block_count += 32

# 	thread_count += 32


# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(c_threads, c_blocks, c_timings)

# ax.set_xlabel('Threads / Block')
# ax.set_ylabel('Blocks')
# ax.set_zlabel('Speed')

# plt.show()