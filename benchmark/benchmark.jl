using CUDA, TropicalNumbers, LinearAlgebra, BenchmarkTools, CuTropicalGEMM, DelimitedFiles

Mat_size = [Int(ceil(10^i)) for i in 1.0:0.5:3.5]

filename = "benchmark.csv"
f = open(filename, "w")

writedlm(f, Mat_size', ',')

# for (MT, T) in [(TropicalMaxPlus, Float32), (TropicalMinPlus, Float32), (TropicalMaxMul, Float32)]
for (MT, T) in [(TropicalMaxPlus, Float32), (TropicalMaxPlus, Float64)]
    cu_time = Vector{Float64}()
    for n in Mat_size
        @show n
        a = MT{T}.(CUDA.rand(T, n, n))
        b = MT{T}.(CUDA.rand(T, n, n))
        o = MT{T}.(CUDA.rand(T, n, n))
        benchmark_result = @benchmark mul!($o, $a, $b);
        time = mean(benchmark_result.times) / 1e9
        push!(cu_time, time)
    end
    writedlm(f, cu_time', ',')
    @show MT, T, cu_time
end

close(f)
println("benchmark end, results saved as benchmark.csv")