#! /bin/bash

# notice that these commands are used to generate the profiling file by Nsight Compute, path of ncu, julia and CuTropicalGemm should be changed accroding to your own machine.

sudo /opt/nvidia/hpc_sdk/Linux_x86_64/23.5/compilers/bin/ncu --set full -f --export report_CuTropical ~/.julia/juliaup/julia-1.9.2+0.x64.linux.gnu/bin/julia benchmark_CuTropicalGemm.jl

