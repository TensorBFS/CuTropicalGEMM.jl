using CUDA
using CuTropicalGEMM

function benchmakr_CuTropicalGemmFP32(m::T, n::T, k::T) where{T}
    A = rand(Float32, (m * k))
    B = rand(Float32, (k * n))
    C = rand(Float32, (m * n))

    CuA = CuArray(A)
    CuB = CuArray(B)
    CuC = CuArray(C)

    MaxAddFP32!(m, n, k, CuA, CuB, CuC)
    MaxMulFP32!(m, n, k, CuA, CuB, CuC)

    return nothing
end

benchmakr_CuTropicalGemmFP32(2560, 2048, 2048)
benchmakr_CuTropicalGemmFP32(2 * 2560, 2 * 2048, 2 * 2048)
benchmakr_CuTropicalGemmFP32(4 * 2560, 4 * 2048, 4 * 2048)