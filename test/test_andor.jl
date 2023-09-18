function direct_andor(A::Matrix{T}, B::Matrix{T}, C::Matrix{T}, i::Int, j::Int) where{T<:Bool}
    K = size(A)[2]

    sum = false

    for k in 1:K
        sum = sum || (A[i, k] && B[k, j])
    end

    sum = sum || C[i, j]

    return sum
end

function check_all_andor(A::Matrix{T}, B::Matrix{T}, C::Matrix{T}, D::Matrix{T}) where{T}
    M = size(A)[1]
    K = size(A)[2]
    N = size(B)[2]
    
    for i in 1:M
        for j in 1:N
            sum = false

            for k in 1:K
                sum = sum || (A[i, k] && B[k, j])
            end

            sum = sum || C[i, j]

            if sum ≉ D[i, j]
                @show i, j, sum, D[i, j]
                return false
            end
        end
    end

    return true
end

@testset "Testing Matrix And Or" begin
    for (M, N, K) in [(5, 6, 7), (101, 102, 103), (1024, 1024, 1024)]
        A = rand(Bool, M, K)
        B = rand(Bool, K, N)
        C = rand(Bool, M, N)

        CuA = CuArray(A)
        CuB = CuArray(B)
        CuC = CuArray(C)

        andor!(CuA, CuB, CuC)

        D = Array(CuC)

        @test check_all_andor(A, B, C, D)
    end
end