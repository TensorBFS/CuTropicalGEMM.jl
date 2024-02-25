# CuTropicalGEMM

[![Build status](https://badge.buildkite.com/06c24dc7b1a9d7c38897acd21575ffd678ee03de190c0b8d81.svg)](https://buildkite.com/julialang/cutropicalgemm-dot-jl)
[![Coverage](https://codecov.io/gh/TensorBFS/CuTropicalGEMM.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/TensorBFS/CuTropicalGEMM.jl)

<p>
CuTropicalGEMM is an open source &nbsp;
    <a href="https://julialang.org">
        <img src="https://raw.githubusercontent.com/JuliaLang/julia-logo-graphics/master/images/julia.ico" width="16em">
        Julia
    </a>
&nbsp; package for fast generic matrix mulplication (GEMM) of tropical numbers on Nvidia GPU base on CUDA.
It greatly speed up the tropical GEMM, which is widely used in tensor network contractions.
</p>

## Features

CuTropicalGEMM support GEMM for various matrix element types:
* and-or algebra: `TropicalAndOr`
* max-plus algebra: `Tropical{Float32/Float64}`
* min-plus algebra: `TropicalMinPlus{Float32/Float64}`
* max-times algebra: `TropicalMaxMul{Float32/Float64/Int32/Int64}`

Please check [`TropicalNumbers.jl`](https://github.com/TensorBFS/TropicalNumbers.jl) for the definitions of these types and semiring algebras. 

## Getting Started

Open a Julia REPL and type `]` to enter the `pkg>` mode, and then install related packages with
```julia
pkg> add CuTropicalGEMM, BenchmarkTools, TropicalNumbers, CUDA
```

Loading `CuTropicalGEMM` module into the workspace affects the `*` and `LinearAlgebra.mul!` on CuTropical matrices immediately. 
The following is a minimum working example:
```julia
julia> using TropicalNumbers, CUDA, BenchmarkTools, LinearAlgebra

julia> a = Tropical.(CUDA.randn(4096, 4096));

julia> @btime CUDA.@sync $a * $a;
  295.465 ms (43 allocations: 1.75 KiB)

julia> using CuTropicalGEMM

julia> @benchmark CUDA.@sync $a * $a
BenchmarkTools.Trial: 442 samples with 1 evaluation.
 Range (min … max):  10.320 ms …  12.313 ms  ┊ GC (min … max): 0.00% … 0.00%
 Time  (median):     11.258 ms               ┊ GC (median):    0.00%
 Time  (mean ± σ):   11.327 ms ± 160.544 μs  ┊ GC (mean ± σ):  0.00% ± 0.00%

                                    █    ▆ ▁  ▃                 
  ▅▁▁▁▄▁▁▁▁▁▁▁▁▁▁▁▄▁▄▁▁▁▁▁▁▁▁▁▄▄▁▁▁▁███▆▁█▆█▅▄█▄▁▁▁▄▁▁▄▁▄▁▄▁▆▆ ▆
  10.3 ms       Histogram: log(frequency) by time      11.9 ms <

 Memory estimate: 272 bytes, allocs estimate: 8.
```

You can also use the function `LinearAlgebra.mul!(o, a, b)`, which allows you to manually allocate memory for the result:

```julia
julia> using LinearAlgebra: mul!

julia> o = Tropical.(CUDA.zeros(4096, 4096));

julia> @benchmark CUDA.@sync mul!($o, $a, $a)
BenchmarkTools.Trial: 440 samples with 1 evaluation.
 Range (min … max):  10.301 ms …  12.117 ms  ┊ GC (min … max): 0.00% … 0.00%
 Time  (median):     11.373 ms               ┊ GC (median):    0.00%
 Time  (mean ± σ):   11.363 ms ± 129.334 μs  ┊ GC (mean ± σ):  0.00% ± 0.00%

                                      ▃    █                    
  ▅▁▁▄▁▁▁▁▁▁▁▁▁▁▁▁▁▄▁▁▁▄▁▁▁▁▁▁▁▁▁▁▁▁▄▁█▄▄▄▄█▇▇▅▄▇▁▁▁▁▄▄▁▁▁▁▅▁▄ ▆
  10.3 ms       Histogram: log(frequency) by time      11.9 ms <

 Memory estimate: 16 bytes, allocs estimate: 1.
```

## Benchmarks

Here is a simple benchmark of the performance using NVIDIA A800 80GB PCIe.
We compared the performance of `CuTropicalGEMM.jl`, `GemmKernels.jl` and direct `CUDA.jl` map reduce on Tropical GEMM with single precision.

The performance of `Cublas` on normal GEMM is used as a reference.

![benchmark FP32](https://github.com/ArrogantGao/CuTropicalGEMM_benchmark/blob/main/plot/benchmark.png)

## Questions and Contributions

Please open an [issue](https://github.com/TensorBFS/CuTropicalGEMM.jl/issues)
if you encounter any problems, or have any feature requests.

If you want to have a check of the `C-CUDA` code, please check the repo [TropicalGemm_Cuda](https://github.com/ArrogantGao/TropicalGemm_Cuda).

It is also welcomed for any suggestions about the issues marked as `enhancement`, please let us know if you have any idea about them.

## Acknowalgement

We would like to thank Tim Besard for his invaluable guidance and support during the development of the package, his expertise in GPU utilization have been immensely helpful. We would also like to thank Tyler Thomas for his assistance in understanding the usage of `BinaryBuilder.jl`.

## References
1. This package originates from the following issue:
https://github.com/JuliaSIMD/LoopVectorization.jl/issues/201
2. When writing our CUDA C package, we referenced the repository https://github.com/Cjkkkk/CUDA_gemm.
