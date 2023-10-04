module CuTropicalGEMM

using CUDA, TropicalNumbers, LinearAlgebra, TropicalGemmC_jll
export matmul!

const path = @__DIR__
const Symbol_FP32 = (:FP32, "FP32")
const Symbol_FP64 = (:FP64, "FP64")
const Symbol_INT32 = (:INT32, "INT32")
const Symbol_INT64 = (:INT64, "INT64")
const Symbol_Bool = (:Bool, "Bool")

const CTranspose{T} = Transpose{T, <:CuVecOrMat{T}}

include("tropical_gemms.jl")

end
