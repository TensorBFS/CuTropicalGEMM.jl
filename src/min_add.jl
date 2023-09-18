export minadd!

function minadd!(A::CuArray{T, 2}, B::CuArray{T, 2}, C::CuArray{T, 2}) where{T <: Number}
    Ar = - A
    Br = - B
    Cr = - C

    maxadd!(Ar, Br, Cr)

    copyto!(C, - Cr)

    return nothing
end