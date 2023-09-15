using CUDA, BenchmarkTools
using LLVMLoopInfo: @loopinfo

# used to store the tilesize as the type of TileSize
struct TileSize{MatSize, BlockSize, ThreadSize} end
# MatSize = (M, N, K)
# BlockSize = (BM, BN, BK)
# ThreadSize = (TM, TN)

@inline function matsize(::Type{TileSize{MatSize, BlockSize, ThreadSize}})::NTuple{3, Int64} where{MatSize, BlockSize, ThreadSize} return MatSize end
@inline function blocksize(::Type{TileSize{MatSize, BlockSize, ThreadSize}})::NTuple{3, Int64} where{MatSize, BlockSize, ThreadSize} return BlockSize end
@inline function threadsize(::Type{TileSize{MatSize, BlockSize, ThreadSize}})::NTuple{2, Int64} where{MatSize, BlockSize, ThreadSize} return ThreadSize end

@inline function offset(row::Int64, col::Int64, h::Int64)::Int64
    return (col - Int64(1)) * h + row
end

function matrix_load(A::CuDeviceMatrix{T, 1}, B::CuDeviceMatrix{T, 1}, C::CuDeviceMatrix{T, 1}, tilesize::Type{TileSize{MatSize, BlockSize, ThreadSize}}) where{T, MatSize, BlockSize, ThreadSize}
    M, N, K = matsize(tilesize)
    BM, BN, BK = blocksize(tilesize)
    TM, TN = threadsize(tilesize)

    bszm = BM ÷ TM
    bszn = BN ÷ TN
    THREAD_NUM_PER_BLOCK = bszm * bszn

    # all the indices are 0-based
    block_id_m = blockIdx().x - 1
    block_id_n = blockIdx().y - 1

    thread_id_m = threadIdx().x - 1
    thread_id_n = threadIdx().y - 1

    grid_dim_m = gridDim().x
    grid_dim_n = gridDim().y

    tid = thread_id_n * bszm + thread_id_m

    # load A and B to shared memory
    A_shmem = CUDA.CuDynamicSharedArray(T, BM * BK)
    B_shmem = CUDA.CuDynamicSharedArray(T, BK * BN, BM * BK * sizeof(T))

    # load C to registers
    C_regs = NTuple{TM * TN, T}(zero(T) for _=1:TM * TN)

    A_TILE_COL = tid ÷ BM
    B_TILE_COL = tid ÷ BN
    
    A_TILE_ROW = tid % BM
    B_TILE_ROW = tid % BN

    A_TILE_COL_STRIDE = THREAD_NUM_PER_BLOCK ÷ BM
    B_TILE_COL_STRIDE = THREAD_NUM_PER_BLOCK ÷ BN

    # start the big iteration
    @loopinfo unroll for tile_idx in 0 : BK : K - 1
        
        # load A into from global memory to shared memory
        @loopinfo unroll for i in 0 : A_TILE_COL_STRIDE : BK - 1
            row = BM * block_id_m + A_TILE_ROW
            col = A_TILE_COL + tile_idx + i

            indice_A_shmem = A_TILE_ROW + (i + A_TILE_COL) * BM + 1
            indice_A = row + col * M + 1

            if block_id_m == grid_dim_m - 1 || block_id_n == grid_dim_n - 1
                A_shmem[indice_A_shmem] = row < M && col < K ? A[indice_A] : zero(T)
            else
                A_shmem[indice_A_shmem] = A[indice_A]
            end
        end

        @loopinfo unrool for i in 0 : B_TILE_COL_STRIDE : BK - 1
            row = BN * block_id_n + B_TILE_ROW
            col = B_TILE_COL + tile_idx + i

            indice_B_shmem = B_TILE_ROW + (i + B_TILE_COL) * BN + 1
            indice_B = row + col * N + 1

            if block_id_m == grid_dim_m - 1 || block_id_n == grid_dim_n - 1
                B_shmem[indice_B_shmem] = row < N && col < K ? B[indice_B] : zero(T)
            else
                B_shmem[indice_B_shmem] = B[indice_B]
            end
        end

        sync_threads()
    end

    return nothing
end



function test()
    M = 4096
    N = 4096
    K = 4096

    BM = 64
    BK = 8
    BN = 64

    TM = 4
    TN = 4

    thread_size = ((BM - 1) ÷ TM + 1, (BN - 1) ÷ TN + 1)
    block_size = ((M - 1) ÷ BM + 1, (N - 1) ÷ BN + 1)

    tilesize = TileSize{(M, N, K), (BM, BN, BK), (TM, TN)}
    
    shmem_size = (BM * BK + BK * BN) * sizeof(Float32)

    A = CUDA.rand(Float32, M, K)
    B = CUDA.rand(Float32, K, N)
    C = CUDA.rand(Float32, M, N)

    BT = permutedims(B, (1, 2))

    # @cuda blocks=block_size threads=thread_size shmem=shmem_size matrix_load(A, B, C, tilesize)

    @btime CUDA.@sync @cuda blocks=$(block_size) threads=$(thread_size) shmem=$(shmem_size) matrix_load($(A), $(BT), $(C), $(tilesize))
end

function indice_check_kernel(thread_id, block_id, grid_dim, tilesize::Type{TileSize{MatSize, BlockSize, ThreadSize}}) where{MatSize, BlockSize, ThreadSize}
    T = Float32
    M, N, K = matsize(tilesize)
    BM, BN, BK = blocksize(tilesize)
    TM, TN = threadsize(tilesize)

    bszm = BM ÷ TM
    bszn = BN ÷ TN
    THREAD_NUM_PER_BLOCK = bszm * bszn

    # all the indices are 0-based
    block_id_m = block_id[1] - 1
    block_id_n = block_id[2] - 1

    thread_id_m = thread_id[1] - 1
    thread_id_n = thread_id[2] - 1

    grid_dim_m = grid_dim[1]
    grid_dim_n = grid_dim[2]

    tid = thread_id_n * bszm + thread_id_m

    # load A and B to shared memory
    # A_shmem = CUDA.CuDynamicSharedArray(T, BM * BK)
    # B_shmem = CUDA.CuDynamicSharedArray(T, BK * BN, BM * BK * sizeof(T))

    # load C to registers
    C_regs = NTuple{TM * TN, T}(zero(T) for _=1:TM * TN)

    A_TILE_COL = tid ÷ BM
    B_TILE_COL = tid ÷ BK
    
    A_TILE_ROW = tid % BM
    B_TILE_ROW = tid % BK

    A_TILE_COL_STRIDE = THREAD_NUM_PER_BLOCK ÷ BM
    B_TILE_COL_STRIDE = THREAD_NUM_PER_BLOCK ÷ BK


    # start the great iteration
    for tile_idx in Int64(0) : BK : K - Int64(1)
        
        # load A into from global memory to shared memory
        # for i in Int64(0) : A_TILE_COL_STRIDE : BK - Int64(1)
        #     row = BM * block_id_m + A_TILE_ROW
        #     col = A_TILE_COL + tile_idx + i
        #     if block_id_m == grid_dim_m - 1 || block_id_n == grid_dim_n - 1
        #         indice_A_shmem = Int64(A_TILE_ROW + (i + A_TILE_COL) * BM + 1)
        #         indice_A = Int64(row + col * M + 1)
        #         if row < M && col < K
        #             @assert indice_A <= M * K
        #         end
        #     else
        #         indice_A_shmem = Int64(A_TILE_ROW + (i + A_TILE_COL) * BM + 1)
        #         indice_A = Int64(row + col * M + 1)
        #     end
        #     # @show indice_A_shmem, indice_A, A_TILE_ROW, i, A_TILE_COL, BM, BK
        #     @assert indice_A_shmem <= BM * BK
        # end

        for i in Int64(0) : B_TILE_COL_STRIDE : BN - Int64(1)
            row = tile_idx + B_TILE_ROW
            col = BN * block_id_n + i + B_TILE_COL
            if block_id_m == grid_dim_m - 1 || block_id_n == grid_dim_n - 1
                indice_B_shmem = Int64(B_TILE_ROW + (i + B_TILE_COL) * BK + 1)
                indice_B = Int64(row + col * K + 1)
                if row < K && col < N
                    @assert indice_B <= K * N
                end
            else
                indice_B_shmem = Int64(B_TILE_ROW + (i + B_TILE_COL) * BK + 1)
                indice_B = Int64(row + col * K + 1)
            end
            @assert indice_B_shmem <= BK * BN
            @show indice_B, indice_B_shmem
        end


    end

    return nothing
end

function indice_check()
    M = 128
    N = 128
    K = 128

    BM = 64
    BK = 16
    BN = 32

    TM = 4
    TN = 4

    threadsize = ((BM - 1) ÷ TM + 1, (BN - 1) ÷ TN + 1)
    blocksize = ((M - 1) ÷ BM + 1, (N - 1) ÷ BN + 1)

    @show threadsize, blocksize

    tilesize = TileSize{(M, N, K), (BM, BN, BK), (TM, TN)}

    for bm in 1:blocksize[1]
        for bn in 1:blocksize[2]
            for tm in 1:threadsize[1]
                for tn in 1:threadsize[2]
                    @show bm, bn, tm, tn
                    indice_check_kernel((tm, tn), (bm, bn), blocksize, tilesize)
                end
            end
        end
    end

    return nothing
end

# indice_check()
test()