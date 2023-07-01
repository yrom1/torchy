#!/bin/bash
CUDA_FILE="hello.cu"
CPP_FILE="grad.cpp"

/usr/local/cuda/bin/nvcc -c $CUDA_FILE -o cuda_obj.o
g++ $CPP_FILE cuda_obj.o -o program -L/usr/local/cuda/lib64 -lcudart
./program
