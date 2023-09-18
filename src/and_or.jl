export andor!

function andor!(A::CuArray{T, 2}, B::CuArray{T, 2}, C::CuArray{T, 2}, M::Int, N::Int, K::Int) where{T <: Bool}

    @ccall libtropicalgemm.BOOL_andor(M::Cint, N::Cint, K::Cint, pointer(A)::CuPtr{Bool}, pointer(B)::CuPtr{Bool}, pointer(C)::CuPtr{Bool})::Cvoid

    return nothing
end