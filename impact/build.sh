/opt/mpich2/osu_ch3_mrail_gen2-intel10/bin/mpicc -o primary_slave.o primary_slave.c -lpthread -lrt -lm -std=c99
gcc -shared -o controller.so -fPIC controller.c -lm -lpthread -lrt -std=c99