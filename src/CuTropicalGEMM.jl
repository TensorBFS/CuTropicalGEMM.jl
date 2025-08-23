module CuTropicalGEMM

using CUDA, LinearAlgebra
using CUDA.CUBLAS
using Reexport
@reexport using TropicalNumbers

export matmul!

# const libtropicalgemm = joinpath(dirname(@__DIR__), "..", "build", "libtropicalgemm.so")
const libtropicalgemm = "/home/xuanzhaogao/code/CuTropicalGEMM/build/libtropicalgemm.so"

const CTranspose{T} = Transpose{T, <:CuVecOrMat{T}}

function dims_match(A::T1, B::T2, C::T3) where{T1, T2, T3}

    @assert size(A, 1) == size(C, 1)
    @assert size(B, 2) == size(C, 2)
    @assert size(A, 2) == size(B, 1)

    return size(A, 1), size(B, 2), size(A, 2)
end

function TropicalNumbers.content(x::T) where{T<:Real}
    return x
end

# wrapper
function cutmsSgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
    CUDA.CUBLAS.initialize_context()
    @ccall libtropicalgemm.cutmsSgemm(handle::CUDA.CUBLAS.cublasHandle_t,
                transa::CUDA.CUBLAS.cublasOperation_t,
                transb::CUDA.CUBLAS.cublasOperation_t, m::Cint, n::Cint,
                k::Cint, alpha::Cfloat,
                A::CuPtr{Cfloat}, lda::Cint,
                B::CuPtr{Cfloat}, ldb::Cint,
                beta::Cfloat, C::CuPtr{Cfloat},
                ldc::Cint)::CUDA.CUBLAS.cublasStatus_t
end

function cutmsDgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
    CUDA.CUBLAS.initialize_context()
    @ccall libtropicalgemm.cutmsDgemm(handle::CUDA.CUBLAS.cublasHandle_t,
                transa::CUDA.CUBLAS.cublasOperation_t,
                transb::CUDA.CUBLAS.cublasOperation_t, m::Cint, n::Cint,
                k::Cint, alpha::Cdouble,
                A::CuPtr{Cdouble}, lda::Cint,
                B::CuPtr{Cdouble}, ldb::Cint,
                beta::Cdouble, C::CuPtr{Cdouble},
                ldc::Cint)::CUDA.CUBLAS.cublasStatus_t
end

for (TA, tA) in [(:CuVecOrMat, 'N'), (:CTranspose, 'T')]
    for (TB, tB) in [(:CuVecOrMat, 'N'), (:CTranspose, 'T')]
        for (TT, CT, funcname) in [
            (:TropicalMaxPlusF32, :Cfloat, :cutmsSgemm), (:TropicalMaxPlusF64, :Cdouble, :cutmsDgemm)
            ]
            @eval function matmul!(C::CuVecOrMat{T}, A::$TA{T}, B::$TB{T}, α::T, β::T) where {T<:$TT}
                M, N, K = dims_match(A, B, C)
                if K == 0 && M * N != 0
                    return rmul!(C, β)
                elseif M * N == 0
                    return C
                else
                    lda = ($tA == 'N') ? K : M
                    ldb = ($tB == 'N') ? N : K
                    ldc = N
                    $funcname(CUDA.CUBLAS.handle(), $tA, $tB, M, N, K, content(α), pointer(parent(A)), lda, pointer(parent(B)), ldb, content(β), pointer(C), ldc)
                end
                return C
            end
        end
    end
end

const CuTropicalBlasTypes = Union{TropicalMaxPlusF32, TropicalMaxPlusF64}

# overload the LinearAlgebra.mul!
for TA in [:CuVecOrMat, :CTranspose]
    for TB in [:CuVecOrMat, :CTranspose]
        @eval function LinearAlgebra.mul!(C::CuVecOrMat{T}, A::$TA{T}, B::$TB{T}, α::Number, β::Number) where {T <: CuTropicalBlasTypes}
            α = _convert(T, α)
            β = _convert(T, β)
            C = matmul!(C, A, B, α, β)
            return C
        end
    end
end

for TT in [:TropicalMaxPlusF32, :TropicalMaxPlusF64]
    @eval _convert(::Type{T}, x::$TT) where T<:$TT = T(x)
    @eval _convert(::Type{T}, x::Number) where T<:$TT = iszero(x) ? zero(T) : (isone(x) ? one(T) : error("Converting from number type `$(typeof(x))` to `$T` is unsafe!"))
end

end
