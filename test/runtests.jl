using CuTropicalGEMM
using Test
using CUDA
using TropicalNumbers
using LinearAlgebra

@testset "CuTropicalGEMM.jl" begin

    include("test_gemms.jl")

    # include("test_maxadd.jl")
    # include("test_minadd.jl")

    # include("test_maxmul.jl")
    # include("test_minmul.jl")
    
    # include("test_andor.jl") 
end
