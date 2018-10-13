mpicc -o primary_slave.o primary_slave.c -lpthread -lrt -lm
gcc -shared -o testlib.so -fPIC test.c -lm -lpthread -lrt
python run.py $1 -np 4 ./primary_slave.o