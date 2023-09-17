using CuTropicalGEMM
using Test
using CUDA

@testset "CuTropicalGEMM.jl" begin
    include("test_maxadd.jl")
    # include("test_maxmul.jl")
    # include("test_minadd.jl")
    # include("test_bool.jl") 
end
