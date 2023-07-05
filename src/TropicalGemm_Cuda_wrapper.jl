export CuTropicalGemmMatmulFP32!

const FP32_lib = joinpath(artifact"CUDA_lib", "TropicalGemmFP32.so")


# A wrapper for FP32 Tropical GEMM
function CuTropicalGemmMatmulFP32!(m::Integer, n::Integer, k::Integer, A::CuArray{Float32}, B::CuArray{Float32}, C::CuArray{Float32})
    A_ptr = pointer(A);
    B_ptr = pointer(B);
    C_ptr = pointer(C);

    @ccall FP32_lib.TropicalMatmul(m::Cint, n::Cint, k::Cint, A_ptr::CuPtr{Cfloat}, B_ptr::CuPtr{Cfloat}, C_ptr::CuPtr{Cfloat})::Cvoid

    return nothing
end