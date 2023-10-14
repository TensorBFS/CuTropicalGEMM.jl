using GenericTensorNetworks, CUDA, TropicalNumbers, GenericTensorNetworks.Graphs
using CuTropicalGEMM
using Random; Random.seed!(2)

# Create a random 3-regular graph
g = GenericTensorNetworks.random_diagonal_coupled_graph(38, 38, 0.8)

# Create a tensor network representation for the independent set problem on this graph
# Let us specify the tensor network contraction order optimizer to be TreeSA, which is a local search algorithm
tn = IndependentSet(g; optimizer=TreeSA(ntrials=1, niters=5))

# Let us check its contraction complexity
contraction_complexity(tn)

# Let us find the maximum independent set using the tensor network contraction.
# It will use the CuTropicalGEMM library to perform the contraction.
# Please use Float32 type for the best performance.
@time Array(solve(tn, SizeMax(); usecuda=true, T=Float32))
# output: 1.1s

# If you want to use the automatic differentiation based approach to find the optimal solution.
@time solve(tn, SingleConfigMax(; bounded=true); usecuda=true, T=Float32)