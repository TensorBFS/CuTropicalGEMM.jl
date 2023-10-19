module CuTropicalGEMM

using CUDA, TropicalNumbers, LinearAlgebra, TropicalGemmC_jll 
export matmul!

function __init__()
    if CUDA.functional() == true
        if CUDA.driver_version() < v"11.4"
            @warn "CUDA.driver_version < v11.4! CuTropicalGEMM may not be available."
        elseif CUDA.driver_version() > v"12.2"
            @warn "CUDA.driver_version > v12.2! CuTropicalGEMM may not be available."
        end
    elseif CUDA.functional() == false
        @warn "CUDA Driver not found! CuTropicalGEMM will not be available."
    end
    return nothing
end

const Symbol_FP32 = (:FP32, "FP32")
const Symbol_FP64 = (:FP64, "FP64")
const Symbol_INT32 = (:INT32, "INT32")
const Symbol_INT64 = (:INT64, "INT64")
const Symbol_Bool = (:Bool, "Bool")

const CTranspose{T} = Transpose{T, <:CuVecOrMat{T}}

include("tropical_gemms.jl")

end
