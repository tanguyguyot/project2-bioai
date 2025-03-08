# expected solution format : list of nurse route's lists i.e. each list is a different nurse's route

using JSON, Random, Plots, Clustering, Base.Threads
include("structures.jl")
include("utilitaries.jl")
include("crossover.jl")
include("initialization.jl")
include("mutations.jl")
include("selections.jl")

train_data0 = JSON.parsefile("./train/test_0.json")
train_data1 = JSON.parsefile("./train/test_1.json")
train_data2 = JSON.parsefile("./train/test_2.json")
train_data9 = JSON.parsefile("./train/train_9.json")

data_instance0 = ProblemInstance(
    train_data0["depot"],
    train_data0["patients"],
    train_data0["nbr_nurses"],
    train_data0["instance_name"],
    train_data0["travel_times"],
    train_data0["benchmark"],
    train_data0["capacity_nurse"],
    train_data0["travel_times"]
)

data_instance1 = ProblemInstance(
    train_data1["depot"],
    train_data1["patients"],
    train_data1["nbr_nurses"],
    train_data1["instance_name"],
    train_data1["travel_times"],
    train_data1["benchmark"],
    train_data1["capacity_nurse"],
    train_data1["travel_times"]
)

data_instance2 = ProblemInstance(
    train_data2["depot"],
    train_data2["patients"],
    train_data2["nbr_nurses"],
    train_data2["instance_name"],
    train_data2["travel_times"],
    train_data2["benchmark"],
    train_data2["capacity_nurse"],
    train_data2["travel_times"]
)

data_instance9 = ProblemInstance(
    train_data9["depot"],
    train_data9["patients"],
    train_data9["nbr_nurses"],
    train_data9["instance_name"],
    train_data9["travel_times"],
    train_data9["benchmark"],
    train_data9["capacity_nurse"],
    train_data9["travel_times"]
)

# GENETIC ALGORITHM 
function genetic_algorithm(instance::ProblemInstance, proportion::Vector{Float64}, num_generations::Int=500, population_size::Int=500, tournament_size::Int=3, child_factor::Int=3, mutation_rate::Float64=0.01, penalty_rate=10.0, custom::Bool=false, populations::Vector{Vector{Individual}}=[[Individual([], 0.0, 0.0, 0.0, false)]])::Tuple{Individual, Vector{Individual}, Vector{Float64}, Vector{Float64}, Vector{Float64}}
    # Initialize a population
    if custom && !isempty(populations[1])
        population = create_population(instance, population_size, proportion, true, populations)
    else
        population = create_population(instance, population_size, proportion)
    end
    nb_nurses::Int = instance.nbr_nurses

    # Keep track of scores and travel times
    scores = []
    feasabilities_record = []
    average_fitnesses = []
    fitnesses = []
    travel_times = []
    feasabilities = []
    feasability_rate = 0.0
    penalty_cost = copy(penalty_rate)
    fitness_cache = Dict{UInt64, Individual}()

    # Initialize best_solution and a counter for redundancy
    best_solution = Individual([], 0.0, 0.0, 0.0, false)
    same_best_solution_counter = 0
    locker = ReentrantLock()
    # Main loop
    for generation in 1:num_generations
        next_population = deepcopy(population)
        if same_best_solution_counter % 5 == 4
            println("Same best solution for more than 5 generations, adding diversity")
            next_population = vcat(next_population, [random_solution(instance) for _ in 1:div(population_size, 2)], [position_cluster_solution(instance) for _ in 1:div(population_size, 2)])
        end
        @threads for _ in 1:(child_factor * population_size)
            # Select two parents
            parent1_idx, parent2_idx = tournament_selection(population, instance, fitness_cache, tournament_size, penalty_cost)
            parent1 = population[parent1_idx]
            parent2 = population[parent2_idx]

            # Crossover
            child1, child2 = crossover(parent1, parent2, instance, fitness_cache, penalty_cost)
            
            # Mutation ; mutation 7 is less likely to happen so we multiply the probability by 1.5
            mutation_rate > rand() ? mutate4!(child1, instance) : nothing
            mutation_rate > rand() ? mutate5!(child1) : nothing
            mutation_rate > rand() ? mutate6!(child1, instance) : nothing
            mutation_rate > rand() * 1.5 ? mutate7!(child1) : nothing
            mutation_rate > rand() ? mutate8!(child1) : nothing
            mutation_rate > rand() ? mutate4!(child2, instance) : nothing
            mutation_rate > rand() ? mutate5!(child2) : nothing
            mutation_rate > rand() ? mutate6!(child2, instance) : nothing 
            mutation_rate > rand() * 1.5 ? mutate7!(child2) : nothing 
            mutation_rate > rand() ? mutate8!(child2) : nothing
            
            # Add to offsprings
            begin
                lock(locker)
                try
                    push!(next_population, child1)
                    push!(next_population, child2)
                finally
                    unlock(locker)
                end
            end
            #push!(next_population, child1)
            #push!(next_population, child2)
        end
        println("End of crossover and mutations for generation $generation")

        # Survivor selection by elitism
        evaluations = [evaluate(sol, instance, fitness_cache, 1.0, max(penalty_cost/10, 0.4), penalty_cost, penalty_cost) for sol in next_population]
        feasabilities = [individual.feasability for individual in evaluations]
        feasability_rate = sum(feasabilities) / length(feasabilities)
        feasability_rate > 0 ? println("Feasibility POSITIVE!! ") : println("Feasibility nulle ")
        penalty_cost = 1 + (penalty_rate - 1)*(1 - feasability_rate)
        population = survivor_selection(evaluations, population_size, tournament_size, penalty_cost)
        # Keep track of scores and average fitnesses
        previous_best_solution = deepcopy(best_solution)
        best_solution = deepcopy(argmin((x -> x.score), population))
        best_solution.routes == previous_best_solution.routes ? same_best_solution_counter += 1 : same_best_solution_counter = 0
        best_solution_score = best_solution.score
        best_solution_travel_time = best_solution.total_travel_time
        fitnesses = [x.score for x in population]
        push!(scores, best_solution_score)
        push!(travel_times, best_solution_travel_time)
        avg_fitness = sum(fitnesses) / length(fitnesses)
        push!(average_fitnesses, avg_fitness)
        print("For generation $generation, best score is $(best_solution_score), travel time of $(best_solution_travel_time), average fitness is $(avg_fitness), feasability rate is $(feasability_rate), penalty cost is $(penalty_cost) \n")
    end
    # Return the best solution
    best_solution = argmin((x -> x.score), population)
    println("Best score found : $(best_solution)\n")
    return best_solution, population, fitnesses, scores, average_fitnesses, travel_times
end
## MAIN ##


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
