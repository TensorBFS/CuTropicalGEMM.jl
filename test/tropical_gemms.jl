@testset "Testing the gemms" begin
    for (MT, DT) in [(TropicalMaxPlus, [Float32, Float64])]
        for T in DT
            for (M, N, K) in [(5, 6, 7), (66, 67, 33), (0, 0, 0), (2, 0, 0), (2, 2, 0)]
                TT = MT{T}
                testset_name = "Type " * string(MT) * "{" * string(T) * "}, size: " * string(M) * " " * string(N) * " " * string(K)
                @testset "$testset_name" begin
                    @info testset_name
                    for TA in ['T', 'N']
                        for TB in ['T', 'N']
                            para_set = [(one(TT), one(TT)), (zero(TT), zero(TT)), (TT(2), TT(3))]
                            for (α, β) in para_set
                                A = TA == 'T' ? transpose(MT.(CuArray(rand(T, K, M)))) : MT.(CuArray(rand(T, M, K)))
                                B = TB == 'T' ? transpose(MT.(CuArray(rand(T, N, K)))) : MT.(CuArray(rand(T, K, N)))
                                C = MT.(CuArray(rand(T, M, N)))

                
                                hA = Array(A)
                                hB = Array(B)
                                hC = Array(C)

                                CUDA.@sync C = CuTropicalGEMM.matmul!(C, A, B, α, β)
                
                                hC .= α .* hA * hB .+ β .* hC

                                @test Array(C) ≈ hC
                            end
                        end
                    end
                end
            end
        end
    end
end

@testset "cuda patch positive and negative" begin
    for (MT, DT) in [(TropicalMaxPlus, [Float32, Float64])]
        for T in DT
            for a in [MT.(CUDA.randn(T, 4, 4)), MT.( - CUDA.randn(T, 4, 4))]
                for b in [MT.(CUDA.randn(T, 4)), MT.( - CUDA.randn(T, 4))]
                    for A in [transpose(a), a, transpose(b)]
                        for B in [transpose(a), a, b]
                            testname = "Type " * string(T) * ", size: " * string(size(A)) * " " * string(size(B))
                            @testset "$testname" begin
                                if !(size(A) == (1,4) && size(B) == (4,))
                                    res0 = Array(A) * Array(B)
                                    CUDA.@sync res1 = A * B
                                    CUDA.@sync res2 = LinearAlgebra.mul!(MT.(CUDA.zeros(T, size(res0)...)), A, B)
                                    @test Array(res1) ≈ res0
                                    @test Array(res2) ≈ res0
                                end
                            end
                        end
                    end
                end
            end
        end
    end
end