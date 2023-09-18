export minmul!

function minmul!(A::CuArray{T, 2}, B::CuArray{T, 2}, C::CuArray{T, 2}, M::Int, N::Int, K::Int) where{T <: Number}
    Ar = - A
    Cr = - C

    maxmul!(Ar, B, Cr, M, N, K)

    copyto!(C, - Cr)

    return nothing
end