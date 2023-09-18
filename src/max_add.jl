export maxadd!

function maxadd!(A::CuArray{T, 2}, B::CuArray{T, 2}, C::CuArray{T, 2}) where{T <: Float32}
    size_A = size(A)
    size_B = size(B)
    size_C = size(C)

    @assert size_A[1] == size_C[1]
    @assert size_A[2] == size_B[1]
    @assert size_B[2] == size_C[2]

    At = permutedims(A, (2, 1));
    Bt = permutedims(B, (2, 1));
    Ct = permutedims(C, (2, 1));

    @ccall libtropicalgemm.FP32_maxadd(size_A[1]::Cint, size_C[2]::Cint, size_A[2]::Cint, pointer(At)::CuPtr{Cfloat}, pointer(Bt)::CuPtr{Cfloat}, pointer(Ct)::CuPtr{Cfloat})::Cvoid

    permutedims!(C, Ct, (2, 1))
    return nothing
end

function maxadd!(A::CuArray{T, 2}, B::CuArray{T, 2}, C::CuArray{T, 2}) where{T <: Float64}
    size_A = size(A)
    size_B = size(B)
    size_C = size(C)

    @assert size_A[1] == size_C[1]
    @assert size_A[2] == size_B[1]
    @assert size_B[2] == size_C[2]

    At = permutedims(A, (2, 1))
    Bt = permutedims(B, (2, 1))
    Ct = permutedims(C, (2, 1))

    @ccall libtropicalgemm.FP64_maxadd(size_A[1]::Cint, size_C[2]::Cint, size_A[2]::Cint, pointer(At)::CuPtr{Cdouble}, pointer(Bt)::CuPtr{Cdouble}, pointer(Ct)::CuPtr{Cdouble})::Cvoid

    permutedims!(C, Ct, (2, 1))

    return nothing
end

# function maxadd!(A::CuArray{T, 2}, B::CuArray{T, 2}, C::CuArray{T, 2}) where{T <: Int32}
#     size_A = size(A)
#     size_B = size(B)
#     size_C = size(C)

#     @assert size_A[1] == size_C[1]
#     @assert size_A[2] == size_B[1]
#     @assert size_B[2] == size_C[2]

#     At = permutedims(A, (2, 1))
#     Bt = permutedims(B, (2, 1))
#     Ct = permutedims(C, (2, 1))

#     @ccall libtropicalgemm.INT32_maxadd(size_A[1]::Cint, size_C[2]::Cint, size_A[2]::Cint, pointer(At)::CuPtr{Cint}, pointer(Bt)::CuPtr{Cint}, pointer(Ct)::CuPtr{Cint})::Cvoid

#     permutedims!(C, Ct, (2, 1))

#     return nothing
# end

# function maxadd!(A::CuArray{T, 2}, B::CuArray{T, 2}, C::CuArray{T, 2}) where{T <: Int64}
#     size_A = size(A)
#     size_B = size(B)
#     size_C = size(C)

#     @assert size_A[1] == size_C[1]
#     @assert size_A[2] == size_B[1]
#     @assert size_B[2] == size_C[2]

#     At = permutedims(A, (2, 1))
#     Bt = permutedims(B, (2, 1))
#     Ct = permutedims(C, (2, 1))

#     @ccall libtropicalgemm.INT64_maxadd(size_A[1]::Cint, size_C[2]::Cint, size_A[2]::Cint, pointer(At)::CuPtr{Clong}, pointer(Bt)::CuPtr{Clong}, pointer(Ct)::CuPtr{Clong})::Cvoid

#     permutedims!(C, Ct, (2, 1))

#     return nothing
# end