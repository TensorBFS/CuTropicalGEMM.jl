export TropicalKernel

struct TropicalKernel
    bm::Int64
    bn::Int64
    bk::Int64
    rm::Int64
    rn::Int64
    enable_prefetch::Bool
end

function matmul(a::T, b::T, c::T) where {T}
    return a * b + c
end

function maxadd(a::T, b::T, c::T) where {T}
    return max(a + b, c)
end


# this kernel function will accept A, B, C as input and update C by the operator function
# M, N and K are the dims of the matrix
# bm, bn, bk are the tile size for submatrix to load in to shared memory (smem)
# bk ≥ TM, TN and bk/TM, bk/TN are integers
# BM and BN are the number of block in x/y dimension
# BK is the number of big iters in blocks
# rm, rn are the tile size for submatrix of A and B in smem
# TM, TN are the size of threads in x/y dimension
# TK is the number of small iteration in threads
function gemm_kernel!(kernel::TropicalKernel, A::CuDeviceMatrix{T,1}, B::CuDeviceMatrix{T,1}, C::CuDeviceMatrix{T,1}, operator::Function) where {T}

    M, K = size(A)
    K, N = size(B)
    bm, bn, bk = kernel.bm, kernel.bn, kernel.bk
    rm, rn = kernel.rm, kernel.rn

    BK = Int(K / bk)
    TM = Int64(kernel.bm / kernel.rm)
    TN = Int64(kernel.bn / kernel.rn)

    # load the tiled matrices into smem
    block_x = blockIdx().x
    block_y = blockIdx().y
    thread_x = threadIdx().x
    thread_y = threadIdx().y

    As = CuDynamicSharedArray(T, (bm, bk))
    Bs = CuDynamicSharedArray(T, (bn, bk), bm * bk * sizeof(T))
    Cs = CuDynamicSharedArray(T, (bm, bn), bm * bk * sizeof(T) + bn * bk * sizeof(T))

    # load C
    for j in 1:rm
        for k in 1:rn
            Cs[(thread_x-1)*rm+j, (thread_y-1)*rn+k] = C[(block_x-1)*bm+(thread_x-1)*rm+j, (block_y-1)*bn+(thread_y-1)*rn+k]
        end
    end

    sync_threads()

    # the big iteration
    for i in 1:BK
        As_per_thread_y = Int(bk / TN)
        Bs_per_thread_x = Int(bk / TM)
        # As_per_thread_y, Bs_per_thread_x are number of elements to load per thread in y/x for As/Bs
        # load A
        for j in 1:rm
            for k in 1:As_per_thread_y
                As[(thread_x-1)*rm+j, (thread_y-1)*As_per_thread_y+k] = A[(block_x-1)*bm+(thread_x-1)*rm+j, (i-1)*bk+(thread_y-1)*As_per_thread_y+k]
            end
        end

        # load B
        for j in 1:Bs_per_thread_x
            for k in 1:rn
                Bs[(thread_y-1)*rn+k, (thread_x-1)*Bs_per_thread_x+j] = B[(i-1)*bk+(thread_x-1)*Bs_per_thread_x+j, (block_y-1)*bn+(thread_y-1)*rn+k]
            end
        end
        # sync threads
        sync_threads()

        # compute
        # for l in 1:bk
        #     for j in 1:rm
        #         for k in 1:rn
        #             Cs[(thread_x - 1) * rm + j, (thread_y - 1) * rn + k] = operator(As[(thread_x - 1) * rm + j, l], Bs[(thread_y - 1) * rn + k, l], Cs[(thread_x - 1) * rm + j, (thread_y - 1) * rn + k])
        #         end
        #     end 
        # end
        # # sync threads
        # sync_threads()
    end

    #write back to C
    # for j in 1:rm
    #     for k in 1:rn
    #         C[(block_x - 1) * bm + (thread_x - 1) * rm + j, (block_y - 1) * bn + (thread_y - 1) * rn + k] = Cs[(thread_x - 1) * rm + j, (thread_y - 1) * rn + k]
    #     end
    # end

    return nothing
end

function Tropical_Gemm!(kernel::TropicalKernel, A::CuArray{T,2}, B::CuArray{T,2}, C::CuArray{T,2}) where {T}
    M, K = size(A)
    Kt, N = size(B)
    Mt, Nt = size(C)
    @assert K == Kt
    @assert M == Mt
    @assert N == Nt

    bm, bn, bk = kernel.bm, kernel.bn, kernel.bk
    rm, rn = kernel.rm, kernel.rn
    BM = Int64(M / kernel.bm)
    BN = Int64(N / kernel.bn)
    TM = Int64(kernel.bm / kernel.rm)
    TN = Int64(kernel.bn / kernel.rn)

    @cuda threads = (TM, TN) blocks = (BM, BN) shmem = (bm * bk + bk * bn + bm * bn) * sizeof(T) gemm_kernel!(kernel, A, B, C, maxadd)

    return nothing
end

function Gemm!(kernel::TropicalKernel, A::CuArray{T,2}, B::CuArray{T,2}, C::CuArray{T,2}) where {T}
    M, K = size(A)
    Kt, N = size(B)
    Mt, Nt = size(C)
    @assert K == Kt
    @assert M == Mt
    @assert N == Nt

    bm, bn, bk, rm, rn = kernel.bm, kernel.bn, kernel.bk, kernel.rm, kernel.rn
    BM = Int64(M / kernel.bm)
    BN = Int64(N / kernel.bn)
    TM = Int64(kernel.bm / kernel.rm)
    TN = Int64(kernel.bn / kernel.rn)

    @cuda threads = (TM, TN) blocks = (BM, BN) shmem = (bm * bk + bk * bn + bm * bn) * sizeof(T) gemm_kernel!(kernel, A, B, C, matmul)

    return nothing
end


begin
    A = rand(Float32, 1024, 1024)
    B = rand(Float32, 1024, 1024)
    C = rand(Float32, 1024, 1024)
    CuA = CuArray(A)
    CuB = CuArray(B)
    CuC = CuArray(C)
    kernel = TropicalKernel(64, 64, 8, 8, 8, false)

    @benchmark CUDA.@sync Gemm!($kernel, $CuA, $CuB, $CuC)
    # Gemm!(kernel, CuA, CuB, CuC);
    # D = C + A * B;
end

function copy_CuArray(A::CuDeviceArray{T,1}) where {T}
    M = size(A)
    thread_x = threadIdx().x

    As = CuDynamicSharedArray(T, M)
    As[thread_x] = A[thread_x]
    sync_threads()
    return nothing
end
