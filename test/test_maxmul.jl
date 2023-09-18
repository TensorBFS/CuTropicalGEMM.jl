# this function will return the (i, j) element of the Tropical result of C + A * B
function direct_maxmul(A::Matrix{T}, B::Matrix{T}, C::Matrix{T}, i::Int, j::Int) where{T}
    K = size(A)[2]

    sum = C[i, j]

    for k in 1:K
        sum = max(sum, A[i, k] * B[k, j])
    end

    return sum
end

function check_all_maxmul(A::Matrix{T}, B::Matrix{T}, C::Matrix{T}, D::Matrix{T}) where{T}
    M = size(A)[1]
    K = size(A)[2]
    N = size(B)[2]
    
    for i in 1:M
        for j in 1:N
            sum = C[i, j]

            for k in 1:K
                sum = max(sum, A[i, k] * B[k, j])
            end

            if sum ≉ D[i, j]
                @show i, j, sum, D[i, j]
                return false
            end
        end
    end

    return true
end

@testset "Testing Matrix max mul" begin
    for (M, N, K) in [(5, 6, 7), (101, 102, 103), (1024, 1024, 1024)]
        @testset "Float32 max mul" begin 
            A = 2f0 .* rand(Float32, M, K)
            B = 2f0 .* rand(Float32, K, N)
            C = 2f0 .* rand(Float32, M, N)

            CuA = CuArray(A)
            CuB = CuArray(B)
            CuC = CuArray(C)

            maxmul!(CuA, CuB, CuC)

            D = Array(CuC)

            @test check_all_maxmul(A, B, C, D)
        end

        @testset "Float64 max mul" begin
            A = 2.0 .* rand(Float64, M, K)
            B = 2.0 .* rand(Float64, K, N)
            C = 2.0 .* rand(Float64, M, N)

            CuA = CuArray(A)
            CuB = CuArray(B)
            CuC = CuArray(C)

            maxmul!(CuA, CuB, CuC)

            D = Array(CuC)

            @test check_all_maxmul(A, B, C, D)
        end

        @testset "Int32 max mul" begin
            A = abs.(rand(Int32, M, K) .% Int32(1000))
            B = abs.(rand(Int32, K, N) .% Int32(1000))
            C = abs.(rand(Int32, M, N) .% Int32(1000))

            CuA = CuArray(A)
            CuB = CuArray(B)
            CuC = CuArray(C)

            maxmul!(CuA, CuB, CuC)

            D = Array(CuC)

            @test check_all_maxmul(A, B, C, D)
        end

        @testset "Int64 max mul" begin
            A = abs.(rand(Int64, M, K) .% Int64(1000))
            B = abs.(rand(Int64, K, N) .% Int64(1000))
            C = abs.(rand(Int64, M, N) .% Int64(1000))

            CuA = CuArray(A)
            CuB = CuArray(B)
            CuC = CuArray(C)

            maxmul!(CuA, CuB, CuC)

            D = Array(CuC)

            @test check_all_maxmul(A, B, C, D)
        end
    end
end