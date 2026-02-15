include("genetic_algo2.jl")
include("plotting.jl")
include("run_experiments.jl")

Random.seed!(1234)

@time begin
    println("Data instance 0")
    sol0_niches, sol0_merge = run_experiment(data_instance0, "output0")
end

@time begin
    println("Data instance 1")
    sol1_niches, sol1_merge = run_experiment(data_instance1, "output1")
end

@time begin
    println("Data instance 2")
    sol2_niches, sol2_merge = run_experiment(data_instance2, "output2")
end
