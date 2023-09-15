using CUDA, BenchmarkTools

function copy_CuMatrix(A::CuDeviceMatrix{T, 1}) where{T}
    M, K = size(A)
    block_x = blockIdx().x
    thread_x = threadIdx().x
    thread_y = threadIdx().y


    As = CuDynamicSharedArray(T, 32, 32)
    As[thread_x, thread_y] = A[(thread_x - 1) * 32 + thread_y, block_x]

    return nothing
end


N = 1024
A = CuArray(rand(Float32, N, N));
@cuda threads = N blocks = N shmem =(N)*sizeof(Float32) copy_CuMatrix(A)

@benchmark CUDA.@sync @cuda threads = (32, 32) blocks = N shmem =(N)*sizeof(Float32) copy_CuMatrix($A)

# compare the speed of copy to matrix and copy to array, A: 1024 * 1024, As: 128 * 8, threads: 16 * 16, block: 8 * 8
function copy_from_matrix_to_array(A::CuDeviceMatrix{T, 1}) where{T}

    block_x = blockIdx().x
    block_y = blockIdx().y
    bid = block_x + (block_y - 1) * 8

    thread_x = threadIdx().x
    thread_y = threadIdx().y
    tid = thread_x + (thread_y - 1) * 16

    As = CuDynamicSharedArray(T, 128 * 8)
    K = Int(1024 / 8)

    num_per_thread = Int(128 * 8 / 16 / 16)

    # large iteration over K
    for i in 1:K
        for j in 1:num_per_thread
            id = (tid - 1) * num_per_thread + j
            A_idx = (block_x - 1) * 128 + (id - 1) % 128 + 1
            A_idy = (i - 1) * 8 + Int64(floor((id - 1)/128)) + 1
            As[id] = A[A_idx, A_idy]
        end
        sync_threads()
    end

    return nothing
end

@inbounds @inline function copy_from_matrix_to_matrix(info::gemm_info{Int64}, A::CuDeviceMatrix{T, 1}, B::CuDeviceMatrix{T, 1}, C::CuDeviceMatrix{T, 1}) where{T}

    block_x = blockIdx().x
    block_y = blockIdx().y
    bid = block_x + (block_y - 1) * info.BM

    thread_x = threadIdx().x
    thread_y = threadIdx().y
    tid = thread_x + (thread_y - 1) * info.TM

    As = CuDynamicSharedArray(T, (info.bm, info.bk))
    Bs = CuDynamicSharedArray(T, (info.bn, info.bk), info.bm * info.bk * sizeof(T))
    Cs = CuDynamicSharedArray(T, (info.bm, info.bn), (info.bm * info.bk + info.bk * info.bn) * sizeof(T))
    
    num_per_thread_A = info.bm * info.bk ÷ (info.TM * info.TN)
    num_per_thread_B = info.bk * info.bn ÷ (info.TM * info.TN)

    # large iteration over 1:BK
    for i in 1:info.BK
        #Load A
        for j in 1:num_per_thread_A
            id = (tid - 1) * num_per_thread_A + j
            As_idx = (id - 1) % info.bm + 1
            As_idy = (id - 1) ÷ info.bm + 1
            A_idx = (block_x - 1) * info.bm + As_idx
            A_idy = (i - 1) * info.bk + As_idy
            As[As_idx, As_idy] = A[A_idx, A_idy]
        end

        #Load B
        for j in 1:num_per_thread_B
            id = (tid - 1) * num_per_thread_B + j
            Bs_idx = (id - 1) % info.bn + 1
            Bs_idy = (id - 1) ÷ info.bn + 1
            B_idy = (block_y - 1) * info.bn + Bs_idx
            B_idx = (i - 1) * info.bk + Bs_idy
            Bs[Bs_idx, Bs_idy] = B[B_idx, B_idy]
        end
        sync_threads()

        #Compute
        for l in 1:info.bk
            for j in 1:info.rm
                for k in 1:info.rn
                    # Cs[(thread_x - 1) * info.rm + j, (thread_y - 1) * info.rn + k] = max(
                    #     Cs[(thread_x - 1) * info.rm + j, (thread_y - 1) * info.rn + k],
                    #     As[(thread_x - 1) * info.rm + j, l] + Bs[(thread_y - 1) * info.rn + k, l]
                    # )
                    Cs[(thread_x - 1) * info.rm + j, (thread_y - 1) * info.rn + k] += As[(thread_x - 1) * info.rm + j, l] * Bs[(thread_y - 1) * info.rn + k, l]
                end
            end 
        end
        sync_threads()
    end

    # write C back
    for i in 1:info.rm
        for j in 1:info.rn
            # C[(block_x - 1) * info.bm + (thread_x - 1) * info.rm + i, (block_y - 1) * info.bn + (thread_y - 1) * info.rn + j] = max(Cs[(thread_x - 1) * info.rm + i, (thread_y - 1) * info.rn + j], C[(block_x - 1) * info.bm + (thread_x - 1) * info.rm + i, (block_y - 1) * info.bn + (thread_y - 1) * info.rn + j])
            C[(block_x - 1) * info.bm + (thread_x - 1) * info.rm + i, (block_y - 1) * info.bn + (thread_y - 1) * info.rn + j] += Cs[(thread_x - 1) * info.rm + i, (thread_y - 1) * info.rn + j]
        end
    end

    return nothing
end

struct gemm_info{T}
    M::T
    N::T
    K::T
    bm::T
    bn::T
    bk::T
    rm::T
    rn::T
    BM::T
    BN::T
    BK::T
    TM::T
    TN::T
end

M, N, K = 2560, 2560, 2560
bm, bn, bk = 64, 64, 8
rm, rn = 4, 4

BM = M ÷ bm
BN = N ÷ bn
BK = K ÷ bk

TM = bm ÷ rm
TN = bn ÷ rn

info = gemm_info{Int64}(M, N, K, bm, bn, bk, rm, rn, BM, BN, BK, TM, TN)

A0 = rand(Float32, M, K);
B0 = rand(Float32, K, N);
C0 = rand(Float32, M, N);

A = CuArray(A0);
B = CuArray(B0);
C = CuArray(C0);

# @cuda blocks = (BM, BN) threads = (TM, TN) shmem=(bm * bk + bn * bk)*sizeof(Float32) copy_from_matrix_to_matrix(info, A, B, C);
# @btime CUDA.@sync @cuda blocks = ($BM, $BN) threads = ($TM, $TN) shmem=$(bm * bk + bn * bk)*sizeof(Float32) copy_from_matrix_to_matrix($info, $A, $B, $C);

@cuda blocks = (BM, BN) threads = (TM, TN) shmem=(bm * bk + bn * bk + bm * bn)*sizeof(Float32) copy_from_matrix_to_matrix(info, A, B, C);
@btime CUDA.@sync @cuda blocks = ($BM, $BN) threads = ($TM, $TN) shmem=$(bm * bk + bn * bk + bm * bn)*sizeof(Float32) copy_from_matrix_to_matrix($info, $A, $B, $C);