#! /bin/bash

nvcc -arch=sm_80 ../src/TropicalSGemmFP32.cu
./a.out

rm a.out