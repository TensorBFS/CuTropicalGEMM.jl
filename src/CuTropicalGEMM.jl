module CuTropicalGEMM

using CUDA, TropicalNumbers, LinearAlgebra
export matmul!

path = @__DIR__
const lib = joinpath(path, "../deps", "TropicalGemmC.so")
const CTranspose{T} = Transpose{T}

include("tropical_gemms.jl")

end
