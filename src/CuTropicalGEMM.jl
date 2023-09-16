module CuTropicalGEMM

using CUDA

const libtropicalgemm = joinpath("/home/xuanzhaogao/code/CuTropicalGEMM.jl", "deps", "TropicalGemmC.so")

include("max_add.jl")
include("min_add.jl")
include("max_mul.jl")
include("and_or.jl")

end
