using CuTropicalGEMM
using Test
using CUDA

@testset "CuTropicalGEMM.jl" begin
    include("test_FP32_matmul.jl")
end
