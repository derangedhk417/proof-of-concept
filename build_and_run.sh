mpicc -o primary_slave.o primary_slave.c -lpthread -lrt
gcc -shared -o testlib.so -fPIC test.c -lm -lpthread -lrt
python run.py -np 8 -hosts 192.168.1.254,192.168.1.236 ./primary_slave.o