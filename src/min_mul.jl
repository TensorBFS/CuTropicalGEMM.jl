export minmul!

function minmul!(A::CuArray{T, 2}, B::CuArray{T, 2}, C::CuArray{T, 2}) where{T <: Number}
    Ar = - A
    Cr = - C

    maxmul!(Ar, B, Cr)

    copyto!(C, - Cr)

    return nothing
end