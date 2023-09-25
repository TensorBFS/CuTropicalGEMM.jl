# Algebra for the following types
# the general API will be given as matmul!(A, B, C)

function dims_match(A::T1, B::T2, C::T3) where{T1, T2, T3}
    MA, KA = size(A)
    KB, NB = size(B)
    MC, NC = size(C)

    @assert MA == MC
    @assert NB == NC
    @assert KA == KB

    return MA, NB, KA
end

for (TA, tA) in [(:CuMatrix, 'N'), (:CTranspose, 'T')]
    for (TB, tB) in [(:CuMatrix, 'N'), (:CTranspose, 'T')]
        for (TT, CT, funcname) in [
            (:Float32, :Cfloat, :FP32_plusmul), (:Float64, :Cdouble, :FP64_plusmul), (:Int32, :Cint, :INT32_plusmul), (:Int64, :Clong, :INT64_plusmul), 
            (:TropicalAndOr, :Bool, :Bool_andor), 
            (:TropicalMaxPlusF32, :Cfloat, :FP32_maxplus), (:TropicalMaxPlusF64, :Cdouble, :FP64_maxplus), 
            (:TropicalMaxMulF32, :Cfloat, :FP32_maxmul), (:TropicalMaxMulF64, :Cdouble, :FP64_maxmul), (:TropicalMaxMulI32, :Cint, :INT32_maxmul), (:TropicalMaxMulI64, :Clong, :INT64_maxmul)]
            @eval function matmul!(A::$TA{T}, B::$TB{T}, C::CuMatrix{T}) where {T<:$TT}
                M, N, K = dims_match(A, B, C)
                if M * N * K == 0
                    return nothing
                else
                    @ccall lib.$funcname(M::Cint, N::Cint, K::Cint, pointer(parent(A))::CuPtr{$CT}, pointer(parent(B))::CuPtr{$CT}, pointer(C)::CuPtr{$CT}, $tA::Cchar, $tB::Cchar)::Cvoid
                end
                return nothing
            end
        end
    end
end

# overload the LinearAlgebra.mul!
for TA in [:CuMatrix, :CTranspose]
    for TB in [:CuMatrix, :CTranspose]
        @eval function LinearAlgebra.mul!(C::CuMatrix{T}, A::$TA{T}, B::$TB{T}) where {T}
            matmul!(A, B, C)
            return C
        end
    end
end