using TropicalNumbers, CUDA, BenchmarkTools

function map_reduce_benchmark(m::T, n::T, k::T) where{T}
    A = Tropical.(CUDA.randn(Float32, (m, k)))
    B = Tropical.(CUDA.randn(Float32, (k, n)))
    C = Tropical.(CUDA.randn(Float32, (k, n)))

    elapsed_time = @belapsed CUDA.@sync begin
        $C = $A * $B
    end

    work_load = 2 * m * n * k
    flops = work_load / elapsed_time / 1e9
    @show m, n, k, elapsed_time, flops
    return nothing
end

map_reduce_benchmark(2560, 2048, 2048)
map_reduce_benchmark(2 * 2560, 2 * 2048, 2 * 2048)
map_reduce_benchmark(4 * 2560, 4 * 2048, 4 * 2048)