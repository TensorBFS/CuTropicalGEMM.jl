using GemmKernels
using GemmKernels.Tiling
using GemmKernels: LocalArray, @immutable
using GemmKernels: Operator
using LLVMLoopInfo: @loopinfo


abstract type TropicalFPUOperation{M, N, K, mb, nb, kb, CT, AT} <: Operator.GeneralFPUOp{M, N, K, mb, nb, kb, CT, AT} end
@inline function Operator.operator_fma(::Type{TropicalFPUOperation{M, N, K, mb, nb, kb, CT, AT}}, a::CT, b::CT, c::AT) where {M, N, K, mb, nb, kb, CT, AT}
    return a * b + c
end

const configs = Dict{}()
@inline function get_Tropical_config(args...)
    val = get(configs, args, nothing)
    if val === nothing
        val = configs[args] = create_Tropical_config(args...)
    end
    return val
end
@noinline function create_Tropical_config(A::Type, sizeA::Dims, stridesA::Dims, transA::Bool,
                                 B::Type, sizeB::Dims, stridesB::Dims, transB::Bool,
                                 C::Type, sizeC::Dims, stridesC::Dims,
                                 alpha::Type, zeroAlpha::Bool,
                                 beta::Type, zeroBeta::Bool,
                                 BM::Int64, BN::Int64, BK::Int64)
    m = sizeA[transA ? 2 : 1]
    k = sizeA[transA ? 1 : 2]
    n = sizeB[transB ? 1 : 2]
    if m != sizeC[1] || n != sizeC[2] || k != sizeB[transB ? 2 : 1]
        throw(DimensionMismatch("Dimensions do not match"))
    end

    a_layout_base = transA ? Layout.RowMajor : Layout.ColMajor
    b_layout_base = transB ? Layout.RowMajor : Layout.ColMajor
    a_aligned_layout_base = transA ? Layout.UnsafeAlignedRowMajor : Layout.UnsafeAlignedColMajor
    b_aligned_layout_base = transB ? Layout.UnsafeAlignedRowMajor : Layout.UnsafeAlignedColMajor

    compute_type = promote_type(eltype(A), eltype(B))
    use_wmma = false

    shared_a_layout = Layout.Padded{a_aligned_layout_base{eltype(A)}, 8}
    shared_b_layout = Layout.Padded{b_aligned_layout_base{eltype(B)}, 8}

    ## outputs are never transposed, and padding them doesn't seem worth it
    shared_c_layout = shared_d_layout = Layout.UnsafeAlignedColMajor{eltype(C)}

    # determine block shape
    # XXX: heuristic should take much more into account (GEMM size, at least)
    block_shape = (M = BM, N = BN, K = BK)

    # determine global memory layouts
    ## check if columns begin at aligned addresses, allowing use of vectorized loads & stores
    a_aligned = (stridesA[2] * sizeof(eltype(A))) % 16 == 0
    b_aligned = (stridesB[2] * sizeof(eltype(B))) % 16 == 0
    c_aligned = (stridesC[2] * sizeof(eltype(C))) % 16 == 0
    ## if alpha is zero, we don't need to load A or B
    if zeroAlpha
        global_a_layout = Layout.Zero{eltype(A)}
        global_b_layout = Layout.Zero{eltype(B)}
    else
        global_a_layout = if a_aligned && m%block_shape.M == 0 &&  k%block_shape.K == 0
            a_aligned_layout_base{eltype(A)}
        else
            a_layout_base{eltype(A)}
        end
        global_b_layout = if b_aligned && k%block_shape.K == 0 && n%block_shape.N == 0
            b_aligned_layout_base{eltype(B)}
        else
            b_layout_base{eltype(B)}
        end
    end
    ## if beta is zero, we don't need to load C
    global_c_layout = if zeroBeta
        Layout.Zero{eltype(C)}
    else
        if c_aligned && m%block_shape.M == 0 && n%block_shape.N == 0
            Layout.UnsafeAlignedColMajor{eltype(C)}
        else
            Layout.ColMajor{eltype(C)}
        end
    end
    global_d_layout = if c_aligned && m%block_shape.M == 0 && n%block_shape.N == 0
        Layout.UnsafeAlignedColMajor{eltype(C)}
    else
        Layout.ColMajor{eltype(C)}
    end

    conf = GemmKernels.get_config(;
            gemm_shape = (M = m, N = n, K = k), block_shape,
            operator = TropicalFPUOperation{8, 8, 1, 4, 8, 1, compute_type, eltype(C)},

            global_a_layout, global_b_layout, global_c_layout, global_d_layout,
            shared_a_layout, shared_b_layout, shared_c_layout, shared_d_layout,

            is_a_col_major = !transA,
            is_b_col_major = !transB
        )

    return conf, compute_type, GemmKernels.kernel(global_a_layout, global_b_layout)
end

function gemm_matmul!(C::CuArray{T}, A::CuArray{T}, B::CuArray{T}, alpha::T, beta::T, BM::TI, BN::TI, BK::TI, transA::Char, transB::Char) where{T, TI <: Integer}

    conf, compute_type, kernel = get_Tropical_config(
        typeof(A), size(A), strides(A), transA=='T',
        typeof(B), size(B), strides(B), transB=='T',
        typeof(C), size(C), strides(C),
        typeof(alpha), iszero(alpha),
        typeof(beta), iszero(beta),
        BM, BN, BK
    )


    GemmKernels.matmul(conf, A, B, C, C; 
        transform_shared_to_regs_a = Transform.Elementwise(x -> x * alpha),
        transform_shared_to_regs_c = Transform.Elementwise(x -> x * beta),
        kernel)

    return C
end

function test()
    M = N = K = 4096
    A = CUDA.rand(Float32, M, K)
    B = CUDA.rand(Float32, K, N)
    C = CUDA.rand(Float32, M, N)
    @info "calculating GEMM for Float32"
    t1 = @elapsed CUDA.@sync gemm_matmul!(C, A, B, 1.0f0, 0.0f0, 128, 128, 32, 'N', 'N')
    @show t1, M * N * K * 2 / t1 / 1e12
    
    TPA = TropicalF32.(A)
    TPB = TropicalF32.(B)
    TPC = TropicalF32.(C)
    @info "calculating GEMM for Tropical MaxPlus Float32"
    t2 = @elapsed CUDA.@sync gemm_matmul!(TPC, TPA, TPB, one(TropicalF32), zero(TropicalF32), 128, 128, 32, 'N', 'N')
    @show t2, M * N * K * 2 / t2 / 1e12

    TPA = TropicalMaxMulF32.(A)
    TPB = TropicalMaxMulF32.(B)
    TPC = TropicalMaxMulF32.(C)
    @info "calculating GEMM for Tropical MaxMul Float32"
    t3 = @elapsed CUDA.@sync gemm_matmul!(TPC, TPA, TPB, one(TropicalMaxMulF32), zero(TropicalMaxMulF32), 128, 128, 32, 'N', 'N')
    @show t3, M * N * K * 2 / t3 / 1e12
    return nothing
end