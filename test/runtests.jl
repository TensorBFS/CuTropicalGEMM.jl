using CuTropicalGEMM
using Test
using CUDA

@testset "CuTropicalGEMM.jl" begin

    include("test_muladd.jl")

    include("test_maxadd.jl")
    include("test_minadd.jl")

    include("test_maxmul.jl")
    include("test_minmul.jl")
    
    include("test_andor.jl") 
end
