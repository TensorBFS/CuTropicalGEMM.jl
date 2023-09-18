export matmul!, kernel_maxmul, kernel_minmul, kernel_maxadd, kernel_minadd, kernel_muladd, kernel_andor

struct TropicalGemmKernel
    operator::Function
end

kernel_maxmul() = TropicalGemmKernel(maxmul!)
kernel_minmul() = TropicalGemmKernel(minmul!)
kernel_maxadd() = TropicalGemmKernel(maxadd!)
kernel_minadd() = TropicalGemmKernel(minadd!)
kernel_muladd() = TropicalGemmKernel(muladd!)
kernel_andor() = TropicalGemmKernel(andor!)

function matmul!(A::CuArray{T, 2}, B::CuArray{T, 2}, C::CuArray{T, 2}, kernel) where{T}
    size_A = size(A)
    size_B = size(B)
    size_C = size(C)

    @assert size_A[1] == size_C[1]
    @assert size_A[2] == size_B[1]
    @assert size_B[2] == size_C[2]

    M, N, K = size_A[1], size_C[2], size_A[2]

    if M * N * K == 0
        return nothing
    else
        At = permutedims(A, (2, 1))
        Bt = permutedims(B, (2, 1))
        Ct = permutedims(C, (2, 1))

        kernel.operator(At, Bt, Ct, M, N, K)

        permutedims!(C, Ct, (2, 1))

        return nothing
    end
end