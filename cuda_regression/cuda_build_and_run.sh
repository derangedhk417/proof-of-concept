nvcc --shared regression.cu -o regression.so --compiler-options -fPIC,-lm,-lpthread,-lrt
