function dims_match(A::T1, B::T2, C::T3) where{T1, T2, T3}

    @assert size(A, 1) == size(C, 1)
    @assert size(B, 2) == size(C, 2)
    @assert size(A, 2) == size(B, 1)

    return size(A, 1), size(B, 2), size(A, 2)
end

function TropicalNumbers.content(x::T) where{T<:Real}
    return x
end

for (TA, tA) in [(:CuVecOrMat, 'N'), (:CTranspose, 'T')]
    for (TB, tB) in [(:CuVecOrMat, 'N'), (:CTranspose, 'T')]
        for (TT, CT, funcname, lib) in [
            (:Float32, :Cfloat, :FLOAT_plusmul, :lib_PlusMul_FP32), (:Float64, :Cdouble, :DOUBLE_plusmul, :lib_PlusMul_FP64), (:Int32, :Cint, :INT_plusmul, :lib_PlusMul_INT32), (:Int64, :Clong, :LONG_plusmul, :lib_PlusMul_INT64), 
            (:TropicalAndOr, :Bool, :BOOL_andor, :lib_TropicalAndOr_Bool), 
            (:TropicalMaxPlusF32, :Cfloat, :FLOAT_maxplus, :lib_TropicalMaxPlus_FP32), (:TropicalMaxPlusF64, :Cdouble, :DOUBLE_maxplus, :lib_TropicalMaxPlus_FP64), 
            (:TropicalMaxMulF32, :Cfloat, :FLOAT_maxmul, :lib_TropicalMaxMul_FP32), (:TropicalMaxMulF64, :Cdouble, :DOUBLE_maxmul, :lib_TropicalMaxMul_FP64), (:TropicalMaxMulI32, :Cint, :INT_maxmul, :lib_TropicalMaxMul_INT32), (:TropicalMaxMulI64, :Clong, :LONG_maxmul, :lib_TropicalMaxMul_INT64)
            ]
            @eval function matmul!(C::CuVecOrMat{T}, A::$TA{T}, B::$TB{T}, α::T, β::T) where {T<:$TT}
                M, N, K = dims_match(A, B, C)
                if K == 0 && M * N != 0
                    return rmul!(C, β)
                elseif M * N == 0
                    return C
                else
                    @ccall $lib.$funcname(M::Cint, N::Cint, K::Cint, pointer(parent(A))::CuPtr{$CT}, pointer(parent(B))::CuPtr{$CT}, pointer(C)::CuPtr{$CT}, content(α)::$CT, content(β)::$CT, $tA::Cchar, $tB::Cchar)::Cvoid
                end
                return C
            end
        end
    end
end

const CuTropicalBlasTypes = Union{TropicalAndOr, TropicalMaxPlusF32, TropicalMaxPlusF64, TropicalMaxMulF32, TropicalMaxMulF64, TropicalMaxMulI32, TropicalMaxMulI64}

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
for TT in [:TropicalMaxMul, :TropicalMaxPlus, :TropicalAndOr, :TropicalMinPlus]
    @eval _convert(::Type{T}, x::$TT) where T<:$TT = T(x)
    @eval _convert(::Type{T}, x::Number) where T<:$TT = iszero(x) ? zero(T) : (isone(x) ? one(T) : error("Converting from number type `$(typeof(x))` to `$T` is unsafe!"))
end
