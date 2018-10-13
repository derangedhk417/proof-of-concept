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

# Here we load the c library that will start the MPI processes
# and interface directly with the rank 0 process of the MPI world.
libname = 'testlib.so'
libdir  = './'
lib     = ctl.load_library(libname, libdir)

# We need to load and configure the initialization function.
lib_init          = lib.init

# Specifies that the first (and only) argument is a (char *)
lib_init.argtypes = [c_char_p]

# Specifies that the return type is int
lib_init.restype  = ctypes.c_int

# Here we construct a string from all of ther arguments
# so that it can be passed into the initialization 
# function.
arg_string = ' '.join(sys.argv[1:])
arg_string = arg_string + '\0' # Make sure to null terminate it
                               # so the c code can handle it.

# Call the libary initialization function.
nProcesses = lib_init(arg_string.encode())

print("[PYTHON] %d processes launched"%nProcesses)

# Load and configure the sendMatrix function from the library.
sendMatrix = lib.sendMatrix

# Specify that the argument is an array of doubles.
sendMatrix.argtypes = [ctl.ndpointer(dtype=ctypes.c_double)]

# Load and confiuger the processArray function.
processArray = lib.processArray
processArray.argtypes = [ctl.ndpointer(dtype=ctypes.c_double)]

# Specify that the return type is an array of double with length
# equal to the number of processes launched minus one. The rank0 
# process just does communication, not processing.
processArray.restype  = ctypes.POINTER(ctypes.c_double*(nProcesses - 1))

print("[PYTHON] Creating matrices")

# Generate a matrix for each rank except for
# zero. Random floats between -10 and 10 are 
# chosen so tha precision is not lost on the
# final sum value returned by the MPI processes.
matrices = []
for i in range(nProcesses - 1):
	mat = []
	for j in range(WIDTH*ROWS):
		mat.append(randf(-10.0, 10.0))
	matrices.append(np.array(mat, np.float64))


print("[PYTHON] Sending matrices")

# Send all of the matrices.
# The primary_slave program will automatically
# assign them to MPI processes.
for i in matrices:
	sendMatrix(i)

inputs  = []

print("[PYTHON] Generating input arrays")

# Generate a list of arrays to pass as inputs
# to the slave program.
for i in range(MSG_COUNT):
	current = []
	for j in range(WIDTH):
		current.append(randf(-10.0, 10.0))
	inputs.append(np.array(current, np.float64))

print("[PYTHON] Processing values")


results = []

# Send each of the input arrays to the slave
# for processing. The timers are used to figure
# out how long execution takes.
start_c = timer()
for i in inputs:
	results.append(processArray(i).contents)
end_c = timer()

print("[PYTHON] Processing complete, verifying data.")

# Now we perform the same calculation on a single 
# thead in Python. This usually takes at least ten 
# times as long as the c part.
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

# Check every result to ensure that it is the same.
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

# Report how long each respective part took.

print("Timing:")
print("\tC      : %fs"%(end_c - start_c))
print("\tPython : %fs"%(end_py - start_py))