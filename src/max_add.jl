export maxadd!

function maxadd!(A::CuArray{T, 2}, B::CuArray{T, 2}, C::CuArray{T, 2}, M::Int, N::Int, K::Int) where{T <: Float32}

    @ccall libtropicalgemm.FP32_maxadd(M::Cint, N::Cint, K::Cint, pointer(A)::CuPtr{Cfloat}, pointer(B)::CuPtr{Cfloat}, pointer(C)::CuPtr{Cfloat})::Cvoid

    return nothing
end

function maxadd!(A::CuArray{T, 2}, B::CuArray{T, 2}, C::CuArray{T, 2}, M::Int, N::Int, K::Int) where{T <: Float64}

    @ccall libtropicalgemm.FP64_maxadd(M::Cint, N::Cint, K::Cint, pointer(A)::CuPtr{Cdouble}, pointer(B)::CuPtr{Cdouble}, pointer(C)::CuPtr{Cdouble})::Cvoid

    return nothing
end

# function maxadd!(A::CuArray{T, 2}, B::CuArray{T, 2}, C::CuArray{T, 2}, M::Int, N::Int, K::Int) where{T <: Int32}

#     @ccall libtropicalgemm.INT32_maxadd(M::Cint, N::Cint, K::Cint, pointer(A)::CuPtr{Cint}, pointer(B)::CuPtr{Cint}, pointer(C)::CuPtr{Cint})::Cvoid

#     return nothing
# end

# function maxadd!(A::CuArray{T, 2}, B::CuArray{T, 2}, C::CuArray{T, 2}, M::Int, N::Int, K::Int) where{T <: Int64}

#     @ccall libtropicalgemm.INT64_maxadd(M::Cint, N::Cint, K::Cint, pointer(A)::CuPtr{Clong}, pointer(B)::CuPtr{Clong}, pointer(C)::CuPtr{Clong})::Cvoid

#     return nothing
# end