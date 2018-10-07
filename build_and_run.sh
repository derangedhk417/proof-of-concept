mpicc -o primary_slave.o primary_slave.c -lpthread -lrt
gcc -shared -o testlib.so -fPIC test.c -lm -lpthread -lrt
python run.py