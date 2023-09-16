export maxmul!

function maxmul!(A::CuArray{T}, B::CuArray{T}, C::CuArray{T}) where{T <: Float32}
    size_A = size(A)
    size_B = size(B)
    size_C = size(C)

    @assert size_A[1] == size_C[1]
    @assert size_A[2] == size_B[1]
    @assert size_B[2] == size_C[2]

    @ccall libtropicalgemm.FP32_maxmul(size_A[1]::Cint, size_C[2]::Cint, size_A[2]::Cint, pointer(A)::CuPtr{Cfloat}, pointer(B)::CuPtr{Cfloat}, pointer(C)::CuPtr{Cfloat})::Cvoid

    return nothing
end

function maxmul!(A::CuArray{T}, B::CuArray{T}, C::CuArray{T}) where{T <: Float64}
    size_A = size(A)
    size_B = size(B)
    size_C = size(C)

    @assert size_A[1] == size_C[1]
    @assert size_A[2] == size_B[1]
    @assert size_B[2] == size_C[2]

    @ccall libtropicalgemm.FP64_maxmul(size_A[1]::Cint, size_C[2]::Cint, size_A[2]::Cint, pointer(A)::CuPtr{Cdouble}, pointer(B)::CuPtr{Cdouble}, pointer(C)::CuPtr{Cdouble})::Cvoid

    return nothing
end

function maxmul!(A::CuArray{T}, B::CuArray{T}, C::CuArray{T}) where{T <: Int32}
    size_A = size(A)
    size_B = size(B)
    size_C = size(C)

    @assert size_A[1] == size_C[1]
    @assert size_A[2] == size_B[1]
    @assert size_B[2] == size_C[2]

    @ccall libtropicalgemm.INT32_maxmul(size_A[1]::Cint, size_C[2]::Cint, size_A[2]::Cint, pointer(A)::CuPtr{Cint}, pointer(B)::CuPtr{Cint}, pointer(C)::CuPtr{Cint})::Cvoid

    return nothing
end

function maxmul!(A::CuArray{T}, B::CuArray{T}, C::CuArray{T}) where{T <: Int64}
    size_A = size(A)
    size_B = size(B)
    size_C = size(C)

    @assert size_A[1] == size_C[1]
    @assert size_A[2] == size_B[1]
    @assert size_B[2] == size_C[2]

    @ccall libtropicalgemm.INT64_maxmul(size_A[1]::Cint, size_C[2]::Cint, size_A[2]::Cint, pointer(A)::CuPtr{Clong}, pointer(B)::CuPtr{Clong}, pointer(C)::CuPtr{Clong})::Cvoid

    return nothing
end