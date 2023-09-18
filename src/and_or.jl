export andor!

function andor!(A::CuArray{T, 2}, B::CuArray{T, 2}, C::CuArray{T, 2}) where{T <: Bool}
    size_A = size(A)
    size_B = size(B)
    size_C = size(C)

    @assert size_A[1] == size_C[1]
    @assert size_A[2] == size_B[1]
    @assert size_B[2] == size_C[2]

    At = permutedims(A, (2, 1))
    Bt = permutedims(B, (2, 1))
    Ct = permutedims(C, (2, 1))

    @ccall libtropicalgemm.BOOL_andor(size_A[1]::Cint, size_C[2]::Cint, size_A[2]::Cint, pointer(At)::CuPtr{Bool}, pointer(Bt)::CuPtr{Bool}, pointer(Ct)::CuPtr{Bool})::Cvoid

    permutedims!(C, Ct, (2, 1))

    return nothing
end