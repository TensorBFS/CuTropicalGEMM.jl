@testset "Testing the gemms" begin
    for (MT, DT) in [(Real, [Float32, Float64, Int32, Int64]), (TropicalAndOr, [Bool]), (TropicalMaxPlus, [Float32, Float64]), (TropicalMaxMul, [Float32, Float64])]
        for T in DT
            for (M, N, K) in [(0, 0, 0), (2, 0, 0), (2, 2, 0), (5, 6, 7), (101, 102, 103), (213, 145, 274)]
                testset_name = "Type " * string(MT) * "{" * string(T) * "}, size: " * string(M) * " " * string(N) * " " * string(K)
                @testset "$testset_name" begin
                    for TA in ['T', 'N']
                        for TB in ['T', 'N']
                            A = TA == 'T' ? transpose(MT.(CuArray(rand(T, K, M)))) : MT.(CuArray(rand(T, M, K)))
                            B = TB == 'T' ? transpose(MT.(CuArray(rand(T, N, K)))) : MT.(CuArray(rand(T, K, N)))
                            C = MT.(CuArray(rand(T, M, N)))
            
                            hA = Array(A)
                            hB = Array(B)
                            hC = Array(C)
            
                            LinearAlgebra.mul!(C, A, B)
            
                            hC .= hA * hB .+ hC

                            if MT <: TropicalAndOr
                                @test Array(C) == hC
                            else
                                @test Array(C) ≈ hC
                            end
                        end
                    end
                end
            end
        end
    end

    for (MT, DT) in [(TropicalMaxMul, [Int32, Int64])]
        for T in DT
            for (M, N, K) in [(0, 0, 0), (2, 0, 0), (2, 2, 0), (5, 6, 7), (101, 102, 103), (213, 145, 274)]
                testset_name = "Type " * string(MT) * "{" * string(T) * "}, size: " * string(M) * " " * string(N) * " " * string(K)
                @testset "$testset_name" begin
                    for TA in ['T', 'N']
                        for TB in ['T', 'N']
                            A = TA == 'T' ? transpose(MT.(CuArray( (abs.(rand(T, K, M))).%1000 ))) : MT.(CuArray( (abs.(rand(T, M, K))).%1000 ))
                            B = TB == 'T' ? transpose(MT.(CuArray( (abs.(rand(T, N, K))).%1000 ))) : MT.(CuArray( (abs.(rand(T, K, N))).%1000 ))
                            C = MT.(CuArray( (abs.(rand(T, M, N))).%1000 ))
            
                            hA = Array(A)
                            hB = Array(B)
                            hC = Array(C)
            
                            LinearAlgebra.mul!(C, A, B)
            
                            hC .= hA * hB .+ hC

                            @test Array(C) ≈ hC
                        end
                    end
                end
            end
        end
    end
end
