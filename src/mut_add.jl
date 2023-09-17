export mutadd!

function mutadd!(A::CuArray{T, 2}, B::CuArray{T, 2}, C::CuArray{T, 2}) where{T <: Float32}
    size_A = size(A)
    size_B = size(B)
    size_C = size(C)

    A = permutedims(A, (2, 1))
    B = permutedims(B, (2, 1))
    C = permutedims(C, (2, 1))

    @assert size_A[1] == size_C[1]
    @assert size_A[2] == size_B[1]
    @assert size_B[2] == size_C[2]

    @ccall libtropicalgemm.FP32_mutadd(size_A[1]::Cint, size_C[2]::Cint, size_A[2]::Cint, pointer(A)::CuPtr{Cfloat}, pointer(B)::CuPtr{Cfloat}, pointer(C)::CuPtr{Cfloat})::Cvoid

    C = permutedims(C, (2, 1))

    return nothing
end

M = 10
N = 15
K = 12

A = 2f0 .* rand(Float32, M, K) .- 1f0
B = 2f0 .* rand(Float32, K, N) .- 1f0
C = 2f0 .* rand(Float32, M, N) .- 1f0

D = C + A * B

CuA = CuArray(A)
CuB = CuArray(B)
CuC = CuArray(C)

At = permutedims(CuA, (2, 1))
Bt = permutedims(CuB, (2, 1))
Ct = permutedims(CuC, (2, 1))


@ccall libtropicalgemm.FP32_mutadd(M::Cint, N::Cint, K::Cint, pointer(At)::CuPtr{Cfloat}, pointer(Bt)::CuPtr{Cfloat}, pointer(Ct)::CuPtr{Cfloat})::Cvoid