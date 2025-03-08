include("genetic_algo2.jl")
Random.seed!(1234)

best_solution9, population9, fitnesses9, scores9, average_fitnesses9 = genetic_algorithm(data_instance9, [0.5, 0.5, 0.0], 500, 1000, 3, 4, 0.1, 10.0)

plot_routes(best_solution9, data_instance9)

# Test data 0
# best_solution0, population0, fitnesses0, scores0, average_fitnesses0 = genetic_algorithm(data_instance0, [0.5, 0.5, 0.0], 500, 1000, 3, 4, 0.1, 10.0)

# Test data 1
# best_solution1, population1, fitnesses1, scores1, average_fitnesses1 = genetic_algorithm(data_instance1, [0.5, 0.5, 0.0], 500, 1000, 3, 4, 0.1, 10.0)

# Test data 2
# best_solution2, population2, fitnesses2, scores2, average_fitnesses2 = genetic_algorithm(data_instance2, [0.5, 0.5, 0.0], 500, 1000, 3, 4, 0.1, 10.0)

# Train data 9
# best_solution9, population9, fitnesses9, scores9, average_fitnesses9 = genetic_algorithm(data_instance9, [0.5, 0.5, 0.0], 500, 1000, 3, 4, 0.1, 10.0)
# Plot functions

# Testing island niching

#=
best_solution91, population91, fitnesses91, scores91, average_fitnesses91 = genetic_algorithm(data_instance9, [1.0, 0, 0.0], 100, 300 , 3, 4, 0.1, 10.0)
best_solution92, population92, fitnesses92, scores92, average_fitnesses92 = genetic_algorithm(data_instance9, [0.0, 1.0, 0.0], 100, 300, 3, 4, 0.1, 10.0)
best_solution93, population93, fitnesses93, scores93, average_fitnesses93 = genetic_algorithm(data_instance9, [0.0, 0.0, 1.0], 100, 300, 3, 4, 0.1, 10.0)

best_solution90, population90, fitnesses90, scores90, average_fitnesses90 = genetic_algorithm(data_instance9, [0.5, 0.5, 0.0], 100, 900, 3, 4, 0.05, 10.0, true, [population91, population92, population93])
BEST : 1562
=# 
