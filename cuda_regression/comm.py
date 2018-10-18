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
def init(directory, name):
	# Here we load the c library that will start the MPI processes
	# and interface directly with the rank 0 process of the MPI world.
	lib = ctl.load_library(name, directory)

	functions = {}

	# We need to load and configure the initialization function.
	configure = lib.configure

	# Specifies that the first (and only) argument is a (char *)
	configure.argtypes = [
		c_float, 
		c_float, 
		c_int, 
		ctl.ndpointer(dtype=c_float),
		c_int,
		c_int
	]

	functions['configure'] = configure
	functions['finish']    = lib.finish

	getRMSE = lib.getRMSE
	getRMSE.argtypes = [ctl.ndpointer(dtype=c_float)]
	getRMSE.restype  = c_float

	functions['getRMSE'] = getRMSE

	return functions