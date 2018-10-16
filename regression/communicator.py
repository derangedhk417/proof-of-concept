import numpy.ctypeslib as ctl
import ctypes
import code
from ctypes import *

# This function takes the directory and name of the library file
# that contains the communication code (written in c). It loads
# the library and its functions. It then calls the init function
# of the library, passing the args parameter to it as a string.
# The function returns a 2-Tuple. The first member is the number
# of MPI processes started and the second parameter is a dictionary
# of imported functions. 
def init(directory, name, args):
	# Here we load the c library that will start the MPI processes
	# and interface directly with the rank 0 process of the MPI world.
	lib = ctl.load_library(name, directory)

	# We need to load and configure the initialization function.
	lib_init          = lib.init

	# Specifies that the first (and only) argument is a (char *)
	lib_init.argtypes = [c_char_p]

	# Specifies that the return type is int
	lib_init.restype  = ctypes.c_int


	# Call the libary initialization function.
	nProcesses = lib_init((args + '\0').encode())

	functions = {}

	cfg = lib.configureDataSize
	cfg.argtypes = [ctypes.c_float, ctypes.c_float, ctypes.c_int]
	functions['configureDataSize'] = cfg

	sendData = lib.sendData
	sendData.argtypes = [ctl.ndpointer(dtype=ctypes.c_double)]
	functions['sendData'] = sendData

	compute = lib.computeRMSE
	compute.argtypes = [ctl.ndpointer(dtype=ctypes.c_double)]
	compute.restype  = ctypes.c_double
	functions['computeRMSE'] = compute

	functions['finish'] = lib.finish

	return (nProcesses, functions)