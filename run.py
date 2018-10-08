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

MSG_COUNT = 50

libname = 'testlib.so'
libdir  = './'
lib     = ctl.load_library(libname, libdir)

lib_init         = lib.init
lib_init.restype = ctypes.c_int
nProcesses = lib_init()

print("%d processes launched"%nProcesses)

sendMatrix = lib.sendMatrix
sendMatrix.argtypes = [ctl.ndpointer(dtype=ctypes.c_double)]

processArray = lib.processArray
processArray.argtypes = [ctl.ndpointer(dtype=ctypes.c_double)]
processArray.restype  = ctypes.POINTER(ctypes.c_double*(nProcesses - 1))

print("Creating matrices")

matrices = []
for i in range(nProcesses - 1):
	mat = []
	for j in range(6000*1024):
		mat.append(randf(-10.0, 10.0))
	matrices.append(np.array(mat, np.float64))


print("Sending matrices")

for i in matrices:
	sendMatrix(i)

inputs  = []

print("Generating input arrays")

for i in range(MSG_COUNT):
	current = []
	for j in range(1024):
		current.append(randf(-10.0, 10.0))
	inputs.append(np.array(current, np.float64))

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

error_count = 0

for i, j in zip(results, expected_results):
	if (list(i) != list(j)):
		print("Computation error with array.")
		print("\tExpected: %s"%str(j))
		print("\tWas:      %s"%str(list(i)))
		error_count += 1

print("All verification complete.")

if (error_count == 0):
	print("No error detected.")
else:
	print("%d erroneous calculations detected.")

print("Timing:")
print("\tC      : %fs"%(end_c - start_c))
print("\tPython : %fs"%(end_py - start_py))