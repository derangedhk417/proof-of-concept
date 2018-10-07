import numpy.ctypeslib as ctl
import numpy as np
import ctypes
import code
import scipy.optimize
from scipy.optimize import curve_fit
from random import randint as rand
from random import uniform as randf
from matplotlib import pyplot as plt
from timeit import default_timer as timer

MSG_COUNT = 10

# compiled library file
libname = 'testlib.so'
libdir  = './'
lib     = ctl.load_library(libname, libdir)

lib_init         = lib.init
lib_init.restype = ctypes.c_int
nProcesses = lib_init()

print("%d processes launched"%nProcesses)

sendMatrix = lib.sendMatrix
sendMatrix.argtypes = [ctl.ndpointer(dtype=ctypes.c_float)]

processArray = lib.processArray
processArray.argtypes = [ctl.ndpointer(dtype=ctypes.c_float)]
processArray.restype  = ctypes.POINTER(ctypes.c_float*(nProcesses - 1))

print("Creating matrices")

matrices = []
for i in range(nProcesses - 1):
	mat = []
	for j in range(6000*1024):
		mat.append(float(randf(-10.0, 10.0)))
	matrices.append(np.array(mat, np.float32))


print("Sending matrices")

for i in matrices:
	sendMatrix(i)

inputs  = []

print("Generating input arrays")

for i in range(MSG_COUNT):
	current = []
	for j in range(1024):
		current.append(float(randf(-10.0, 10.0)))
	inputs.append(np.array(current, np.float32))

print("Processing values")


results = []

start_c = timer()
for i in inputs:
	results.append(processArray(i).contents)
end_c = timer()

print("Processing complete, verifying data.")


expected_results = []
start_py = timer()
for ip in inputs:
	ip_results = []
	for matrix in matrices:
		result = 0.0
		for j in range(6000):
			for k in range(1024):
				result += matrix[j*1024 + k] * ip[k]
		ip_results.append(result)
	expected_results.append(ip_results)
end_py = timer()

print("Verification computations done. Comparing.")

for i, j in zip(results, expected_results):
	if (list(i) != list(j)):
		print("Computation error with array.")
		print("\tExpected: %s"%str(j))
		print("\tWas:      %s"%str(list(i)))
		break

print("All verification complete.")

print("C took %fs"%(end_c - start_c))
print("Python took %fs"%(end_py - start_py))

# V_LJ = lib.LennardJonesPotential
# V_LJ.argtypes = [ctypes.c_double, ctypes.c_double, ctypes.c_double]
# V_LJ.restype  = ctypes.c_double

# # We need to determine the domain we are evaluating the function over.
# # The length of this array is necessary when specifying the length
# # of the return array for the C implementation of the function.
# x = np.arange(1.0, 5, 0.05)


# V_LJ_Array = lib.LennardJonesPotentialIntegralArray

# # This is how argument types are specified.
# V_LJ_Array.argtypes = [
# 	ctl.ndpointer(dtype=ctypes.c_double),
# 	ctypes.c_double, ctypes.c_double, ctypes.c_int
# ]

# # This is how the return type is specified.
# V_LJ_Array.restype = ctypes.POINTER(ctypes.c_double*len(x))


# # Python implementation of the LJ potential.
# def LennardJonesPotential(r, epsilon, sigma):
# 	return 4.0 * epsilon * (np.power(sigma / r, 12) - np.power(sigma / r, 6))

# # Python implrementation of the integral of the LJ potential.
# # This is a very basic rectangular approximation of the integral.
# def LennardJonesPotentialIntegral(r, epsilon, sigma):
# 	out = [0.0]*len(r)
# 	for (i, idx) in zip(r, range(len(r))):
# 		R = 1.0
# 		result = 0.0
# 		while (R < i):
# 			result += 0.001 * LennardJonesPotential(R, epsilon, sigma)
# 			R += 0.001
# 		out[idx] = result

# 	return out

# # Used to generate the imperfect data to regress over.
# def RandFloatSmall():
# 	return rand(0, 100) / 800.0

# # This wrapper calls the actual C function and passes the length
# # of the input array as a parameter. This is necessary because the
# # scipy optimization functions do not do this.
# def CallC_Impl_Array(r, epsilon, sigma):
# 	return np.array(V_LJ_Array(r, epsilon, sigma, len(r)).contents)

# # Generate integrated LJ potential with some noise.
# y = [LennardJonesPotentialIntegral([i], 1, 1)[0] + RandFloatSmall() for i in x]

# # Plot the noisy LJ potential integral.
# fig = plt.figure(1)
# ax  = fig.add_subplot(111)
# plt.plot(x, y)

# major_ticks_x = np.arange(0.0, 5.0, 0.5)
# minor_ticks_x = np.arange(0.0, 5.0, 0.1)

# ax.set_xticks(major_ticks_x)
# ax.set_xticks(minor_ticks_x, minor=True)

# major_ticks_y = np.arange(-2.0, 8.0, 1.0)
# minor_ticks_y = np.arange(-2.0, 8.0, 0.2)

# ax.set_yticks(major_ticks_y)
# ax.set_yticks(minor_ticks_y, minor=True)

# ax.grid(which='both', linestyle=':')
# ax.grid(which='minor', alpha=0.4)
# ax.grid(which='major', alpha=0.6)

# plt.ylim(-1.0, 2.0)
# plt.xlim(0.5, 5.0)
# plt.xlabel("r")
# plt.ylabel(r"Lennard-Jones Potential $\int_1^r{V_{LJ}(r) dr}$")
# plt.title(r"$\int_1^r{V_{LJ}(r) dr}$ with small Random Error")

# plt.show()

# print("Running Regessions . . . this may take a while")



# # Time the regression using the C function.
# start_c = timer()

# res_c, cov_c = curve_fit(CallC_Impl_Array, x, y)
# err_c        = np.sqrt(np.diag(cov_c))

# end_c = timer()


# # Time the regression using the Python function.
# start_py = timer()

# res_py, cov_py = curve_fit(LennardJonesPotentialIntegral, x, y)
# err_py        = np.sqrt(np.diag(cov_py))

# end_py = timer()


# # Display results.

# print("C Implementation:")
# print("\tEpsilon Error: %f"%err_c[0])
# print("\tSigma Error:   %f"%err_c[1])
# print("\tElapsed Time:  %fs\n\n"%(end_c - start_c))

# print("Python Implementation:")
# print("\tEpsilon Error: %f"%err_py[0])
# print("\tSigma Error:   %f"%err_py[1])
# print("\tElapsed Time:  %fs\n\n"%(end_py - start_py))

# print("C implementation was %f times faster"%((end_py - start_py) / (end_c - start_c)))

# # Interpolate over the domain using the parameters returned by the regression.
# c_impl_v =  [LennardJonesPotentialIntegral([i], res_c[0], res_c[1])[0] for i in x]
# py_impl_v = [LennardJonesPotentialIntegral([i], res_py[0], res_py[1])[0] for i in x]


# # Plot the resulting regressions.
# fig = plt.figure(2)
# ax  = fig.add_subplot(111)
# line,    = plt.plot(x, y, color='black')
# c_impl,  = plt.plot(x, c_impl_v, linestyle='dashed', color='red')
# py_impl, = plt.plot(x, py_impl_v, linestyle='dotted', color='green')

# major_ticks_x = np.arange(0.0, 5.0, 0.5)
# minor_ticks_x = np.arange(0.0, 5.0, 0.1)

# ax.set_xticks(major_ticks_x)
# ax.set_xticks(minor_ticks_x, minor=True)

# major_ticks_y = np.arange(-2.0, 8.0, 1.0)
# minor_ticks_y = np.arange(-2.0, 8.0, 0.2)

# ax.set_yticks(major_ticks_y)
# ax.set_yticks(minor_ticks_y, minor=True)

# ax.grid(which='both', linestyle=':')
# ax.grid(which='minor', alpha=0.4)
# ax.grid(which='major', alpha=0.6)

# plt.ylim(-1.0, 2.0)
# plt.xlim(0.5, 5.0)
# plt.xlabel("r")
# plt.ylabel(r"Lennard-Jones Potential $\int_1^r{V_{LJ}(r) dr}$")
# plt.legend(
# 	[line, c_impl, py_impl], 
# 	["Potential With Error", "C Regression", "Python Regression"]
# )
# plt.title(r"$\int_1^r{V_{LJ}(r) dr}$ With Error and Regressions")

# plt.show()

#code.interact(local=locals())