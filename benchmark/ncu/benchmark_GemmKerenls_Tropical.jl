# notice: this code is used to benchmark the Tropical matmul in GemmKernels.jl, which is not yet released in the latest version and only supported in julia@v1.7
# to run the code, you need to manually download the latest version repo of GemmKernels.jl and activate the enviroment

import CUDA

using CUDA
using GemmKernels
using LinearAlgebra

CUDA.allowscalar(false)

function try_tropical(M, N, K)
    for (A_type, B_type, CD_type, min_dimension) in [(Float32, Float32, Float32, 128)], 
        transpose_a = [true, false], 
        transpose_b = [true, false], 
        (OP_M, OP_N, OP_K) in [(8, 16, 2)]

        a_h = rand(A_type, (M, K)) / sqrt(A_type(K))
        b_h = rand(B_type, (K, N)) / sqrt(B_type(K))
        c_h = rand(CD_type, (M, N))
        d_h = similar(c_h)

        
        # Transpose input if necessary
        a_h = transpose_a ? transpose(a_h) : a_h
        b_h = transpose_b ? transpose(b_h) : b_h

        a   = CuArray(a_h)
        b   = CuArray(b_h)
        c   = CuArray(c_h)
        d   = similar(c)

        conf = GemmKernels.get_config(
                                        gemm_shape = (M = M, N = N, K = K),
                                        block_shape = (M = 64, N = 64, K = 32),
                                        operator = Operator.TropicalFPUOp{OP_M, OP_N, OP_K, CD_type, A_type},
                                        global_a_layout = transpose_a ? Layout.AlignedRowMajor{A_type} : Layout.AlignedColMajor{A_type},
                                        global_b_layout = transpose_b ? Layout.AlignedRowMajor{B_type} : Layout.AlignedColMajor{B_type},

                                        global_c_layout = Layout.AlignedColMajor{CD_type},
                                        global_d_layout = Layout.AlignedColMajor{CD_type},

                                        is_a_col_major = !transpose_a,
                                        is_b_col_major = !transpose_b,
                                        )

        GemmKernels.matmul(a, b, c, d, conf; kernel = Kernel.matmul_pipelined) 
    end
    return nothing
end


try_tropical(2560, 2048, 2048)
try_tropical(2 * 2560, 2 * 2048, 2 * 2048)
try_tropical(4 * 2560, 4 * 2048, 4 * 2048)