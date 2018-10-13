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
import communicator


nProcesses, functions = communicator.init('./', 'testlib.so', ' '.join(sys.argv[1:]))

