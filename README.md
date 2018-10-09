# proof-of-concept
python controlled mpi POC

This code exists to prove that the following can be done:

1) Load a C library in Python
2) C library launches a set of MPI processes
3) C library initializes IPC between the rank0 MPI process and itself
4) Python calls C library to make it do calculations
5) C library in turn offloads calculations onto MPI processes
6) Each MPI process other thank rank0 keeps a large amount of data stored in memory that it uses for processing smaller chunks of data sent in by the rank0 process.
7) Results of all computations are sent back to Python for processing
