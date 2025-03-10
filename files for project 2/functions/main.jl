include("genetic_algo2.jl")
include("plotting.jl")

Random.seed!(1234)

@time begin
    println("Data instance 0, niche 1")
    sol0_1 = genetic_algorithm(data_instance0, [0.3, 0.7, 0], 60, 100, 3, 4, 0.15, 300.0, "output0_1")
    println("Data instance 0, niche 2")
    sol0_2 = genetic_algorithm(data_instance0, [0.7, 0.3, 0], 60, 100, 3, 4, 0.15, 300.0, "output0_2")
    println("Data instance 0, niche 3")
    sol0_3 = genetic_algorithm(data_instance0, [0.0, 0.0, 1.0], 60, 100, 3, 4, 0.15, 300.0, "output0_3")
    println("Data instance 0, island merge")
    sol0 = genetic_algorithm(data_instance0, [0.5, 0.5, 0], 30, 300, 3, 4, 0.15, 500.0, "output0", true, [sol0_1[3], sol0_2[3], sol0_3[3]])
end

output_solution(sol0_1[1], data_instance0, "sol0_1")
output_solution(sol0_2[1], data_instance0, "sol0_2")
output_solution(sol0_3[1], data_instance0, "sol0_3")
output_solution(sol0[1], data_instance0, "sol0")
output_solution(sol0[2], data_instance0, "sol0_feasible")
output_all_solution(sol0, "output01")

# output plots
plot_travels_over_time(sol0_1[7], "output0_1")
plot_travels_over_time(sol0_2[7], "output0_2")
plot_travels_over_time(sol0_3[7], "output0_3")
plot_travels_over_time(sol0[7], "output0")
plot_best_solution_over_time(sol0_1[8], "output0_1")
plot_best_solution_over_time(sol0_2[8], "output0_2")
plot_best_solution_over_time(sol0_3[8], "output0_3")
plot_best_solution_over_time(sol0[8], "output0")

@time begin
    println("Data instance 1, niche 1")
    @time sol1_1 = genetic_algorithm(data_instance1, [0.3, 0.7, 0], 60, 100, 3, 4, 0.15, 300.0, "output1_1")
    println("Data instance 1, niche 2")
    @time sol1_2 = genetic_algorithm(data_instance1, [0.7, 0.3, 0], 60, 100, 3, 4, 0.15, 300.0, "output1_2")
    println("Data instance 1, niche 3")
    @time sol1_3 = genetic_algorithm(data_instance1, [0.0, 0.0, 1.0], 60, 100, 3, 4, 0.15, 300.0, "output1_3")
    @time sol1 = genetic_algorithm(data_instance1, [0.5, 0.5, 0], 30, 300, 3, 4, 0.15, 500.0, "output1", true, [sol1_1[3], sol1_2[3], sol1_3[3]])
end

output_solution(sol1_1[1], data_instance0, "sol1_1")
output_solution(sol1_2[1], data_instance0, "sol1_2")
output_solution(sol1_3[1], data_instance0, "sol1_3")
output_solution(sol1[1], data_instance0, "sol1")
output_solution(sol1[2], data_instance0, "sol1_feasible")

#output plots
plot_travels_over_time(sol1_1[7], "output1_1")
plot_travels_over_time(sol1_2[7], "output1_2")
plot_travels_over_time(sol1_3[7], "output1_3")
plot_travels_over_time(sol1[7], "output1")
plot_best_solution_over_time(sol1_1[8], "output1_1")
plot_best_solution_over_time(sol1_2[8], "output1_2")
plot_best_solution_over_time(sol1_3[8], "output1_3")
plot_best_solution_over_time(sol1[8], "output1")

@time begin
    println("Data instance 2, niche 1")
    @time sol2_1 = genetic_algorithm(data_instance2, [0.3, 0.7, 0], 60, 100, 3, 4, 0.15, 300.0, "output2_1")
    println("Data instance 2, niche 2")
    @time sol2_2 = genetic_algorithm(data_instance2, [0.7, 0.3, 0], 60, 100, 3, 4, 0.15, 300.0, "output2_2")
    println("Data instance 2, niche 3")
    @time sol2_3 = genetic_algorithm(data_instance2, [0.0, 0.0, 1.0], 60, 100, 3, 4, 0.15, 300.0, "output2_3")
    println("Data instance 2, island merge")
    @time sol2 = genetic_algorithm(data_instance2, [0.5, 0.5, 0], 30, 300, 3, 4, 0.15, 500.0, "output2", true, [sol2_1[3], sol2_2[3], sol2_3[3]])
end

output_solution(sol2_1[1], data_instance0, "sol2_1")
output_solution(sol2_2[1], data_instance0, "sol2_2")
output_solution(sol2_3[1], data_instance0, "sol2_3")
output_solution(sol2[1], data_instance0, "sol2")
output_solution(sol2[2], data_instance0, "sol2_feasible")
#output_all_solution(sol2, "output21")

# output plots

plot_travels_over_time(sol2_1[7], "output2_1")
plot_travels_over_time(sol2_2[7], "output2_2")
plot_travels_over_time(sol2_3[7], "output2_3")
plot_travels_over_time(sol2[7], "output2")
plot_best_solution_over_time(sol2_1[8], "output2_1")
plot_best_solution_over_time(sol2_2[8], "output2_2")
plot_best_solution_over_time(sol2_3[8], "output2_3")
plot_best_solution_over_time(sol2[8], "output2")
=#