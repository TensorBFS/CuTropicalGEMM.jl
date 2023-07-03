using CUDA, Test, BenchmarkTools, Dates

function CuTropicalGemmMatmul!(m::Integer, n::Integer, k::Integer, A::CuArray{Float32}, B::CuArray{Float32}, C::CuArray{Float32})
    A_ptr = pointer(A);
    B_ptr = pointer(B);
    C_ptr = pointer(C);

    @ccall "./TropicalSGemm.so".TropicalMatmul(m::Cint, n::Cint, k::Cint, A_ptr::CuPtr{Cfloat}, B_ptr::CuPtr{Cfloat}, C_ptr::CuPtr{Cfloat})::Cvoid
    return nothing
end

# this function will return the (i, j) element of the Tropical result of C + A * B
function TropicalDirectMatmul_ij(i::Integer, j::Integer, m::Integer, n::Integer, k::Integer, A::Array{Float32}, B::Array{Float32}, C::Array{Float32})
    sum = zero(Float32)
    for l in 1:k
        sum = max(A[(i - 1) * k + l] + B[(l - 1) * n + j], sum)
    end
    return sum
end

#init the system
m, n, k = 1024, 1024, 1024

A = rand(Float32, (m * k));
B = rand(Float32, (k * n));
C = rand(Float32, (m * n));

CuA = CuArray(A);
CuB = CuArray(B);
CuC = CuArray(C);

CuTropicalGemmMatmul!(m, n, k, CuA, CuB, CuC)

C_result = Array(CuC);


# test
@testset begin 
    for _ in 1:40
        for _ in 1:40
            a = rand(1:m)
            b = rand(1:n)
            @test TropicalDirectMatmul_ij(a, b, m, n, k, A, B, C) ≈ C_result[(a - 1) * n + b]
        end
    end
end