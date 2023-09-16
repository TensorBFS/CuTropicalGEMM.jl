function direct_andor(A::Matrix{T}, B::Matrix{T}, C::Matrix{T}, i::Int, j::Int) where{T<:Bool}
    K = size(A)[2]

    sum = false

    for k in 1:K
        sum = sum || (A[i, k] && B[k, j])
    end

    sum = sum || C[i, j]

    return sum
end

@testset "Testing Matrix And Or" begin
    M = rand(3000:6000)
    N = rand(3000:6000)
    K = rand(3000:6000)
    A = rand(Bool, M, K)
    B = rand(Bool, K, N)
    C = rand(Bool, M, N)

    CuA = CuArray(A)
    CuB = CuArray(B)
    CuC = CuArray(C)

    andor!(CuA, CuB, CuC)

    D = Array(CuC)

    for _ in 1:100
        i = rand(1:M)
        j = rand(1:N)
        @test D[i, j] ≈ direct_maxadd(A, B, C, i, j)
    end
end