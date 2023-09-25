# this function will return the (i, j) element of the Tropical result of C + A * B
function direct_mutadd(A::Matrix{T}, B::Matrix{T}, C::Matrix{T}, i::Int, j::Int) where{T}
    K = size(A)[2]

    sum = C[i, j]

    for k in 1:K
        sum += A[i, k] * B[k, j]
    end

    return sum
end

@testset "Testing Matrix mul add" begin

    kernel = kernel_muladd()
    for (M, N, K) in [(0, 0, 0), (2, 0, 0), (2, 2, 0), (5, 6, 7), (101, 102, 103), (1024, 1024, 1024)]
        @testset "Float32 mul add" begin 
            A = 2f0 .* rand(Float32, M, K) .- 1f0
            B = 2f0 .* rand(Float32, K, N) .- 1f0
            C = 2f0 .* rand(Float32, M, N) .- 1f0

            CuA = CuArray(A)
            CuB = CuArray(B)
            CuC = CuArray(C)

            matmul!(CuA, CuB, CuC, kernel)

            D = Array(CuC)

            @test D ≈ C + A * B
            # @test D[M, N] ≈ direct_mutadd(A, B, C, M, N)
        end

        @testset "Float64 mul add" begin
            A = 2.0 .* rand(Float64, M, K) .- 1.0
            B = 2.0 .* rand(Float64, K, N) .- 1.0
            C = 2.0 .* rand(Float64, M, N) .- 1.0

            CuA = CuArray(A)
            CuB = CuArray(B)
            CuC = CuArray(C)

            matmul!(CuA, CuB, CuC, kernel)

            D = Array(CuC)

            @test D ≈ C + A * B
            # @test D[M, N] ≈ direct_mutadd(A, B, C, M, N)
        end

        @testset "Int32 mul add" begin
            A = rand(Int32, M, K) .% Int32(1000)
            B = rand(Int32, K, N) .% Int32(1000)
            C = rand(Int32, M, N) .% Int32(1000)

            CuA = CuArray(A)
            CuB = CuArray(B)
            CuC = CuArray(C)

            matmul!(CuA, CuB, CuC, kernel)

            D = Array(CuC)

            @test D ≈ C + A * B
            # @test D[M, N] ≈ direct_mutadd(A, B, C, M, N)
        end

        @testset "Int64 mul add" begin
            A = rand(Int64, M, K) .% Int64(1000)
            B = rand(Int64, K, N) .% Int64(1000)
            C = rand(Int64, M, N) .% Int64(1000)
            
            CuA = CuArray(A)
            CuB = CuArray(B)
            CuC = CuArray(C)

            matmul!(CuA, CuB, CuC, kernel)

            D = Array(CuC)

            @test D ≈ C + A * B
            # @test D[M, N] ≈ direct_mutadd(A, B, C, M, N)
        end
    end
end