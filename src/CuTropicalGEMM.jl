module CuTropicalGEMM

using CUDA, TropicalNumbers, LinearAlgebra
export matmul!

const path = @__DIR__
const Symbol_FP32 = (:FP32, "FP32")
const Symbol_FP64 = (:FP64, "FP64")
const Symbol_INT32 = (:INT32, "INT32")
const Symbol_INT64 = (:INT64, "INT64")
const Symbol_Bool = (:Bool, "Bool")

for (Algebra, SAlgebra, Symbol_types) in [(:PlusMul, "PlusMul", [Symbol_FP32, Symbol_FP64, Symbol_INT32, Symbol_INT64]), (:TropicalAndOr, "TropicalAndOr", [Symbol_Bool]), (:TropicalMaxMul, "TropicalMaxMul", [Symbol_FP32, Symbol_FP64, Symbol_INT32, Symbol_INT64]), (:TropicalMaxPlus, "TropicalMaxPlus", [Symbol_FP32, Symbol_FP64])]
    for Symbol_type in Symbol_types
        t, st = Symbol_type
        @eval const $(Symbol("lib_$(Algebra)_$(t)")) = joinpath(path, "../deps/lib", "lib_" * $SAlgebra * "_" * $st * ".so")
    end
end

const CTranspose{T} = Transpose{T, <:CuVecOrMat{T}}

include("tropical_gemms.jl")

end
