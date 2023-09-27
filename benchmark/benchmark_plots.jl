using DelimitedFiles, Plots

# filename = "benchmark.csv"
# file = open(filename, "r")

# benchmark_results = []

M_size = [10,32,100,317,1000,3163]
FP32_time = [6.781416988888889e-6,6.3605972500000005e-6,1.633231715e-5,3.496373941e-5,0.00021173563968657352,0.005052347612]
FP64_time = [1.2943315944444445e-5,1.344461096e-5,4.397673236e-5,0.00010117792614948349,0.0008792887499121265,0.02305435885]

plot(dpi = 500)
plot!(log10.(M_size), log10.(FP32_time), label = "FP32")
plot!(log10.(M_size), log10.(FP64_time), label = "FP64")
xlabel!("log(n)")
ylabel!("log(t) / (seconds)")
title!("time cost of Tropical operations by CuTropicalGEMM.jl")
savefig("time.png")

# OP = 2 .* M_size.^3
# FP32_TFLOPS = OP ./ FP32_time ./ 1e12
# FP64_TFLOPS = OP ./ FP64_time ./ 1e12

# plot(dpi = 500)
# plot!(log10.(M_size), FP32_TFLOPS, label = "FP32")
# plot!(log10.(M_size), FP64_TFLOPS, label = "FP64")
# xlabel!("log(n)")
# ylabel!("TFLOPS")
# title!("TFLOPS Tropical operations by CuTropicalGEMM on A800.jl")
# savefig("tflops.png")