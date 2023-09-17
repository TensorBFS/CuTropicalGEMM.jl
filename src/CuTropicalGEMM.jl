module CuTropicalGEMM

using CUDA

path = @__DIR__
const libtropicalgemm = joinpath(path, "../deps", "TropicalGemmC.so")

# include("max_add.jl")
# include("min_add.jl")
# include("max_mul.jl")
# include("and_or.jl")
include("mut_add.jl")

end
