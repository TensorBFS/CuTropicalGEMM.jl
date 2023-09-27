@testset "Testing the gemms" begin
    for (MT, DT) in [(Real, [Float32, Float64, Int32, Int64]), (TropicalAndOr, [Bool]), (TropicalMaxPlus, [Float32, Float64]), (TropicalMaxMul, [Float32, Float64]), (TropicalMaxMul, [Int32, Int64])]
        for T in DT
            for (M, N, K) in [(0, 0, 0), (2, 0, 0), (2, 2, 0), (5, 6, 7), (101, 102, 103)]
                if MT == Real
                    TT = T
                elseif MT == TropicalAndOr
                    TT = MT
                else
                    TT = MT{T}
                end
                testset_name = "Type " * string(MT) * "{" * string(T) * "}, size: " * string(M) * " " * string(N) * " " * string(K)
                @testset "$testset_name" begin
                    for TA in ['T', 'N']
                        for TB in ['T', 'N']
                            para_set = !(MT == TropicalAndOr) ? [(one(TT), one(TT)), (zero(TT), zero(TT)), (TT(2), TT(3))] : [(one(TT), one(TT)), (zero(TT), zero(TT)), (TT(true), TT(false))]
                            for (α, β) in para_set
                                if MT == TropicalMaxMul
                                    A = TA == 'T' ? transpose(MT.(CuArray( T.((abs.(rand(T, K, M))).% 1000) ))) : MT.(CuArray( T.((abs.(rand(T, M, K))).% (1000))))
                                    B = TB == 'T' ? transpose(MT.(CuArray( T.((abs.(rand(T, N, K))).% 1000) ))) : MT.(CuArray( T.((abs.(rand(T, K, N))).% (1000))))
                                    C = MT.(CuArray( T.((abs.(rand(T, M, N))).%1000)))
                                else
                                    A = TA == 'T' ? transpose(MT.(CuArray(rand(T, K, M)))) : MT.(CuArray(rand(T, M, K)))
                                    B = TB == 'T' ? transpose(MT.(CuArray(rand(T, N, K)))) : MT.(CuArray(rand(T, K, N)))
                                    C = MT.(CuArray(rand(T, M, N)))
                                end

                
                                hA = Array(A)
                                hB = Array(B)
                                hC = Array(C)

                                C = CuTropicalGEMM.matmul!(C, A, B, α, β)
                
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

@testset "cuda patch" begin
    for (MT, DT) in [(Real, [Float32, Float64, Int32, Int64]), (TropicalAndOr, [Bool]), (TropicalMaxPlus, [Float32, Float64]), (TropicalMaxMul, [Float32, Float64])]
        for T in DT
            a = MT.(CUDA.rand(T, 4, 4))
            b = MT.(CUDA.rand(T, 4))
            for A in [transpose(a), a, transpose(b)]
                for B in [transpose(a), a, b]
                    testname = "Type " * string(T) * ", size: " * string(size(A)) * " " * string(size(B))
                    @testset "$testname" begin
                        if !(size(A) == (1,4) && size(B) == (4,))
                            res0 = Array(A) * Array(B)
                            res1 = A * B
                            res2 = LinearAlgebra.mul!(MT.(CUDA.zeros(T, size(res0)...)), A, B)
                            @test Array(res1) ≈ res0
                            @test Array(res2) ≈ res0
                        end
                    end
                end
            end
        end
    end
end

# test over negative values.
@testset "cuda patch negative" begin
    for T in [Tropical{Float64}]
        a = T.(-CUDA.rand(4, 4))
        b = T.(-CUDA.rand(4))
        for A in [transpose(a), a, transpose(b)]
            for B in [transpose(a), a, b]
                if !(size(A) == (1,4) && size(B) == (4,))
                    @info typeof(A), typeof(B)
                    res0 = Array(A) * Array(B)
                    res1 = A * B
                    res2 = LinearAlgebra.mul!(CUDA.zeros(T, size(res0)...), A, B, true, false)
                    @test Array(res1) ≈ res0
                    @test Array(res2) ≈ res0
                end
            end
        end
    end
end