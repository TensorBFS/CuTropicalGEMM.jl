using CuTropicalGEMM
using Test
using CUDA
using TropicalNumbers
using LinearAlgebra

@testset "CuTropicalGEMM.jl" begin

    include("tropical_gemms.jl")

end
