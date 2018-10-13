mpicc -o primary_slave.o primary_slave.c -lpthread -lrt -lm
gcc -shared -o testlib.so -fPIC test.c -lm -lpthread -lrt
python run.py -np 4 ./primary_slave.o