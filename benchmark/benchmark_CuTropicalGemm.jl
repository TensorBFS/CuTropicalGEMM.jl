using CUDA
using BenchmarkTools
using CuTropicalGEMM

function benchmakr_CuTropicalGemmFP32(m::T, n::T, k::T) where{T}
    A = rand(Float32, (m * k))
    B = rand(Float32, (k * n))
    C = rand(Float32, (m * n))

    CuA = CuArray(A)
    CuB = CuArray(B)
    CuC = CuArray(C)

    # I found hat @belapsed and CUDA.@sync can not properly benchmark the function from .so lib
    elapsed_time = @belapsed CUDA.@sync begin
        1 + 1
        CuTropicalGemmMatmulFP32!($m, $n, $k, $CuA, $CuB, $CuC)
        1 + 1
    end

    work_load = 2 * m * n * k
    flops = work_load / elapsed_time / 1e9
    @show m, n, k, elapsed_time, flops
    return nothing
end

benchmakr_CuTropicalGemmFP32(2560, 2048, 2048)
benchmakr_CuTropicalGemmFP32(2 * 2560, 2 * 2048, 2 * 2048)
benchmakr_CuTropicalGemmFP32(4 * 2560, 4 * 2048, 4 * 2048)