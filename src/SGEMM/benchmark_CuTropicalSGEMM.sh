#! /bin/bash

nvcc -arch=sm_80 TropicalSGemm.cu
./a.out

rm a.out