import numpy.ctypeslib as ctl
import numpy as np
import ctypes
from ctypes import *
import code
import scipy.optimize
import sys
from scipy.optimize import curve_fit
from random import randint as rand
from random import uniform as randf
from matplotlib import pyplot as plt
from timeit import default_timer as timer

MSG_COUNT = 5
WIDTH     = 1024
ROWS      = 6000

libname = 'testlib.so'
libdir  = './'
lib     = ctl.load_library(libname, libdir)

arg_string = ' '.join(sys.argv[1:])

lib_init          = lib.init
lib_init.argtypes = [c_char_p]
lib_init.restype  = ctypes.c_int

arg_string = arg_string + '\0'

nProcesses = lib_init(arg_string.encode())

print("[PYTHON] %d processes launched"%nProcesses)

sendMatrix = lib.sendMatrix
sendMatrix.argtypes = [ctl.ndpointer(dtype=ctypes.c_double)]

processArray = lib.processArray
processArray.argtypes = [ctl.ndpointer(dtype=ctypes.c_double)]
processArray.restype  = ctypes.POINTER(ctypes.c_double*(nProcesses - 1))

print("[PYTHON] Creating matrices")

matrices = []
for i in range(nProcesses - 1):
	mat = []
	for j in range(WIDTH*ROWS):
		mat.append(randf(-10.0, 10.0))
	matrices.append(np.array(mat, np.float64))


print("[PYTHON] Sending matrices")

for i in matrices:
	sendMatrix(i)

inputs  = []

print("[PYTHON] Generating input arrays")

for i in range(MSG_COUNT):
	current = []
	for j in range(WIDTH):
		current.append(randf(-10.0, 10.0))
	inputs.append(np.array(current, np.float64))

print("[PYTHON] Processing values")


results = []

start_c = timer()
for i in inputs:
	results.append(processArray(i).contents)
end_c = timer()

print("[PYTHON] Processing complete, verifying data.")


expected_results = []
start_py = timer()
for ip in inputs:
	ip_results = []
	for matrix in matrices:
		result = 0.0
		for j in range(ROWS):
			for k in range(WIDTH):
				result += matrix[j*WIDTH + k] * ip[k]
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