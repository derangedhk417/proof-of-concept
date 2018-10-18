nvcc --shared regression.cu -o regression.so --compiler-options -shared,-fPIC,-lm,-lpthread,-lrt
