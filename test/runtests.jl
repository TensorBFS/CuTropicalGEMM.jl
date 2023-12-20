using CuTropicalGEMM
using Test
using CUDA
using LinearAlgebra

@testset "CuTropicalGEMM.jl" begin

    include("tropical_gemms.jl")

end
