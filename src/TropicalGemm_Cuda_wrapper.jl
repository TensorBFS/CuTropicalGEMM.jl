export MaxAddFP32!, MaxMulFP32!

const MaxAddFP32_lib = joinpath(artifact"CuTropicalGemm_lib", "TropicalGemmMaxAddFP32.so")
const MaxMulFP32_lib = joinpath(artifact"CuTropicalGemm_lib", "TropicalGemmMaxMulFP32.so")

# A wrapper for FP32 Tropical GEMM
function MaxAddFP32!(m::Integer, n::Integer, k::Integer, A::CuArray{Float32}, B::CuArray{Float32}, C::CuArray{Float32})
    A_ptr = pointer(A);
    B_ptr = pointer(B);
    C_ptr = pointer(C);

    @ccall MaxAddFP32_lib.TropicalGemmMaxAdd(m::Cint, n::Cint, k::Cint, A_ptr::CuPtr{Cfloat}, B_ptr::CuPtr{Cfloat}, C_ptr::CuPtr{Cfloat})::Cvoid

    return nothing
end

function MaxMulFP32!(m::Integer, n::Integer, k::Integer, A::CuArray{Float32}, B::CuArray{Float32}, C::CuArray{Float32})
    A_ptr = pointer(A);
    B_ptr = pointer(B);
    C_ptr = pointer(C);

    @ccall MaxMulFP32_lib.TropicalGemmMaxMul(m::Cint, n::Cint, k::Cint, A_ptr::CuPtr{Cfloat}, B_ptr::CuPtr{Cfloat}, C_ptr::CuPtr{Cfloat})::Cvoid

    return nothing
end