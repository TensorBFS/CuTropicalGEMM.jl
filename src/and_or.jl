export andor!

# this one is quiet confusing

function andor!(A::CuArray{T}, B::CuArray{T}, C::CuArray{T}) where{T <: Bool}
    size_A = size(A)
    size_B = size(B)
    size_C = size(C)

    @assert size_A[1] == size_C[1]
    @assert size_A[2] == size_B[1]
    @assert size_B[2] == size_C[2]

    @ccall libtropicalgemm.Bool_andor(size_A[1]::Cint, size_C[2]::Cint, size_A[2]::Cint, pointer(A)::CuPtr{Bool}, pointer(B)::CuPtr{Bool}, pointer(C)::CuPtr{Bool})::Cvoid

    return nothing
end