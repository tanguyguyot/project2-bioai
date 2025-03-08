# expected solution format : list of nurse route's lists i.e. each list is a different nurse's route

using JSON, Random, Plots, Clustering, Base.Threads
include("structures.jl")

train_data0 = JSON.parsefile("train/test_0.json")
train_data1 = JSON.parsefile("train/test_1.json")
train_data2 = JSON.parsefile("train/test_2.json")
train_data9 = JSON.parsefile("train/train_9.json")

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

# PRACTICAL FUNCTIONS

function get_travel_time(i::Int, j::Int, instance::ProblemInstance)
    travel_time_matrix = instance.travel_time_matrix
    # adds +1 because of the depot which is 0
    return travel_time_matrix[i+1][j+1]
end

# Inserts a patient to the best neighbour according to evaluate function
function insert_to_best_neighbour!(patient::Int64, individual::Individual, instance::ProblemInstance, fitness_cache::Dict{UInt64, Individual}, penalty_cost::Float64=350.0)
    routes = individual.routes
    lowest_score = Inf
    best_neighbour_idx = 0
    best_route_idx = 0
    for (route_idx, route) in enumerate(routes)
        for position in 1:length(route)+1
            candidate_route = copy(route)
            # Insert patient in the route
            insert!(candidate_route, position, patient)
            # check in hashmap
            individual_hash = hash_individual(individual)
            score = haskey(fitness_cache, individual_hash) ? score = fitness_cache[individual_hash].score : evaluate(Individual([candidate_route], 0.0, 0.0, 0.0, false), instance, fitness_cache, 1.0, max(div(penalty_cost, 10), 1.0), penalty_cost, penalty_cost).score
            if score < lowest_score
                lowest_score = score
                best_neighbour_idx = position
                best_route_idx = route_idx
            end
        end
    end
    insert!(individual.routes[best_route_idx], best_neighbour_idx, patient)
end

# Inserts a patient to the best neighbours ; also checks if have capacity
function insert_to_closest_neighbour!(patient::Int64, individual::Individual, instance::ProblemInstance)
    routes = individual.routes
    lowest_distance = Inf
    best_neighbour = 0
    best_neighbour_idx = 0
    best_route = []
    best_route_idx = 0
    patient_demand = instance.patients[string(patient)]["demand"]
    capacity = instance.capacity_nurse
    # Reduce routes to check to the ones that have enough capacity ; normally there'll always be at least one
    for (route_idx, route) in enumerate(routes)
        for (route_patient_idx, route_patient) in enumerate(route)
            # get distance between patient to insert and all the other patients
            distance = get_travel_time(patient, route_patient, instance)
            if distance < lowest_distance
                lowest_distance = distance
                best_neighbour = route_patient
                best_neighbour_idx = route_patient_idx
                best_route = route
                best_route_idx = route_idx
            end
        end
    end

    # Special case : if the best neighbour is the only patient in the route
    if (best_neighbour_idx == 1) && (length(best_route) == 1)
        insert!(individual.routes[best_route_idx], 2, patient)
        return
    # Best neighbour is the first of a several-patients route : check whether depot is closer than next patient
    elseif best_neighbour_idx == 1
        # Place at the beginning of the route
        get_travel_time(0, patient, instance) < get_travel_time(patient, best_route[2], instance) ? insert!(individual.routes[best_route_idx], 1, patient) : insert!(individual.routes[best_route_idx], 2, patient)
        return
    # Best neighbour is the last of a several-patients route : check whether depot is closer than previous patient
    elseif best_neighbour_idx == length(best_route)
        # Place at the end of the route
        get_travel_time(best_route[end-1], patient, instance) < get_travel_time(patient, 0, instance) ? insert!(individual.routes[best_route_idx], best_neighbour_idx, patient) : push!(individual.routes[best_route_idx], patient)
        return
    else
    # Place between the best neighbour and the second closest that surrounds the best neighbour 
        patient_before = best_route[best_neighbour_idx-1]
        patient_after = best_route[best_neighbour_idx+1]
        get_travel_time(patient_before, patient, instance) < get_travel_time(patient, patient_after, instance) ? insert!(individual.routes[best_route_idx], best_neighbour_idx, patient) : insert!(individual.routes[best_route_idx], best_neighbour_idx+1, patient)
        return
    end
end

# INITIALIZATION

# Cluster patients based on their position

function position_cluster_solution(instance::ProblemInstance)::Individual
    nb_patients = length(instance.patients)
    nb_nurses = instance.nbr_nurses
    coordinates = hcat([instance.patients[string(patient)]["x_coord"] for patient in 1:nb_patients], [instance.patients[string(patient)]["y_coord"] for patient in 1:nb_patients])'
    R = kmeans(coordinates, nb_nurses)
    clusters = R.assignments
    routes = [[] for _ in 1:nb_nurses]
    for cluster in 1:nb_nurses
        patients = findall(x -> x == cluster, clusters)
        routes[cluster] = patients
    end
    return Individual(routes, 0.0, 0.0, 0.0, false)
end

# Cluster patients based on their time window : later...

function time_window_cluster_solution(instance::ProblemInstance)::Individual
    patients = instance.patients
    nb_patients = length(patients)
    nb_nurses = instance.nbr_nurses # nb de clusters
    data = [ [(patients[string(patient)]["start_time"] + patients[string(patient)]["end_time"]) / 2, patients[string(patient)]["end_time"] - patients[string(patient)]["start_time"]] for patient in 1:nb_patients]
    data_matrix = hcat(data...)
    R = kmeans(data_matrix, nb_nurses)
    clusters = R.assignments
    clusters_and_idx = [(i, clusters[i]) for i in eachindex(clusters)]
    routes = [[] for _ in 1:nb_nurses]
    for route in routes
        # pour chaque route on ajoute chaque cluster 1 fois de 1 à 25
        for cluster in 1:nb_nurses
            idx = findfirst(x -> x[2] == cluster, clusters_and_idx)
            if isnothing(idx)
                continue
            end
            push!(route, clusters_and_idx[idx][1])
            deleteat!(clusters_and_idx, idx)
        end
        sort!(route, by = x -> (instance.patients[string(x)]["start_time"],
                                instance.patients[string(x)]["end_time"]))
    end
    return Individual(routes, 0.0, 0.0, 0.0, false)
end

# Completely random solution
function random_solution(instance::ProblemInstance)::Individual
    nb_patients = length(instance.patients)
    nb_nurses = instance.nbr_nurses
    permutation = randperm(nb_patients)
    separations = sort(sample(1:nb_patients, nb_nurses-1, replace=false))
    finale_routes = [[] for _ in 1:nb_nurses]
    route_counter = 1 # de 1 à 25
    for i in 1:nb_patients
        if route_counter == 25 #last route
            for j in i:nb_patients
            push!(finale_routes[route_counter], permutation[j])
            end
            break
        end
        if i == separations[route_counter]
            route_counter += 1
        end
        push!(finale_routes[route_counter], permutation[i])
    end
    return Individual(finale_routes, 0.0, 0.0, 0.0, false)
end

# Create a population of solutions, based on the initialization functions

function create_population(instance::ProblemInstance, population_size::Int, proportion::Vector{Float64}, custom::Bool=false, populations::Vector{Vector{Individual}}=[[Individual([], 0.0, 0.0, 0.0, false)]])::Vector{Individual}
    if custom
        println("yes la pop")
        return vcat(populations[1], populations[2], populations[3])
    end
    return vcat([random_solution(instance) for _ in 1:(population_size * proportion[1])], 
                [position_cluster_solution(instance) for _ in 1:population_size * proportion[2]],
                [time_window_cluster_solution(instance) for _ in 1:population_size * proportion[3]])
end

# EVALUATION FUNCTION
function hash_individual(individual::Individual)::UInt64
    return hash(individual.routes)
end

function evaluate(individual::Individual, instance::ProblemInstance, fitness_cache::Dict{UInt64, Individual}, unit_cost::Float64=1.0, delay_cost=2.0, capacity_cost=50.0, late_depot_cost=50.0)::Individual
    individual_hash = hash_individual(individual)
    if haskey(fitness_cache, individual_hash)
        return fitness_cache[individual_hash]
    end
    
    total_cost::Float64 = 0
    total_distance::Float64 = 0
    total_penalty::Float64 = 0
    # Find the best route splitting
    feasability = true
    capacity = instance.capacity_nurse
    routes = individual.routes
    deadline = instance.depot["return_time"]
    for (idx_route, route) in enumerate(routes)
        route_distance = 0
        elapsed_time = 0
        last_patient_id = 0
        nurse_demand = 0
        penalty_cost = 0
        for patient_id in route
            # Calculate section distance
            distance = get_travel_time(last_patient_id, patient_id, instance)
            # Update route_distance
            route_distance += distance
            # Calculate time cost
            arrival_time = elapsed_time + distance
            wait_time = max(instance.patients[string(patient_id)]["start_time"] - arrival_time, 0)
            
            # Update elapsed time, because late time takes care into account
            elapsed_time = arrival_time + wait_time + instance.patients[string(patient_id)]["care_time"] 
            late_time = max(elapsed_time - instance.patients[string(patient_id)]["end_time"], 0)
            
            # no wait time cost because he will just wait for the patient's time window to open
            penalty_cost += delay_cost * late_time
            
            # Update last patient ID
            last_patient_id = patient_id
            # Update nurse demand
            nurse_demand += instance.patients[string(patient_id)]["demand"]
        end
        # Capacity penalty
        nurse_demand > capacity ? total_cost += capacity_cost : nothing
        # Calculate back to depot cost
        route_distance += get_travel_time(last_patient_id, 0, instance)
        # Calculate back to depot time cost
        elapsed_time += get_travel_time(last_patient_id, 0, instance)
        # Calculate late depot cost penalty
        late_depot_time = max(elapsed_time - deadline, 0)
        # Penalty cost is the penalty cost for the route
        penalty_cost += late_depot_time * late_depot_cost
        # Calculate route cost : normal distance + penalties
        route_cost = unit_cost * route_distance + penalty_cost
        # Update total cost and total distance
        total_cost += route_cost
        total_distance += unit_cost * route_distance
        total_penalty += penalty_cost
    end
    if total_penalty > 0
        feasability = false
    end
    evaluated_individual = Individual(individual.routes, total_distance, total_cost, total_penalty, feasability)
    fitness_cache[individual_hash] = evaluated_individual
    return evaluated_individual
end

# PARENT SELECTION

# A simple tournament selection function for parent selection
function tournament_selection(population::Vector{Individual}, instance::ProblemInstance, fitness_cache::Dict{UInt64, Individual}, k::Int=3, penalty_cost::Float64=10.0)::Tuple{Int, Int}
    idx_candidates1 = rand(1:length(population), k)
    candidates1 = [population[i] for i in idx_candidates1]
    evaluated_candidates1 = [evaluate(x, instance, fitness_cache, 1.0, max(div(penalty_cost, 10), 1.0), penalty_cost, penalty_cost) for x in candidates1]
    min_idx_score1 = argmin([x.score for x in evaluated_candidates1])
    
    idx_candidates2 = rand(setdiff(1:length(population), [idx_candidates1[min_idx_score1]]), k)
    candidates2 = [population[i] for i in idx_candidates2]
    evaluated_candidates2 = [evaluate(x, instance, fitness_cache, 1.0, max(div(penalty_cost, 10), 1.0), penalty_cost, penalty_cost) for x in candidates2]
    min_idx_score2 = argmin([x.score for x in evaluated_candidates2])
    
    return idx_candidates1[min_idx_score1], idx_candidates2[min_idx_score2]
end

# CROSSOVER FUNCTION 

# Crossover function, partially matched crossover
function crossover(parent1::Individual, parent2::Individual, instance::ProblemInstance, fitness_cache::Dict{UInt64, Individual}, penalty_cost::Float64=350.0)::Tuple{Individual, Individual}
    # convert into list of routes to identify routes
    routes1 = parent1.routes
    routes2 = parent2.routes
    available_routes1 = [i for i in 1:length(routes1) if !isempty(routes1[i])]
    available_routes2 = [i for i in 1:length(routes2) if !isempty(routes2[i])]

    # initialize childs
    child1::Individual = deepcopy(parent1)
    child2::Individual = deepcopy(parent2)

    # Select two random nurses' routes from each parent
    route1_idx::Int64 = rand(available_routes1)
    route2_idx::Int64 = rand(available_routes2)
    route1 = routes1[route1_idx]
    route2 = routes2[route2_idx]

    # remove the patients of route1 from the first parent and vice versa
    for patient in route2
        for (route_idx, _) in enumerate(routes1)
            idx = findfirst(x -> x == patient, child1.routes[route_idx])
            isnothing(idx) ? continue : deleteat!(child1.routes[route_idx], idx) |> break
        end
    end
    for patient in route1
        for (route_idx, _) in enumerate(routes2)
            idx = findfirst(x -> x == patient, child2.routes[route_idx])
            isnothing(idx) ? continue : deleteat!(child2.routes[route_idx], idx) |> break
        end
    end

    # now for each patient without a visitor, find the best insertion
    for patient in route2
        insert_to_best_neighbour!(patient, child1, instance, fitness_cache, penalty_cost)
    end
    for patient in route1
        insert_to_best_neighbour!(patient, child2, instance, fitness_cache, penalty_cost)
    end
    return child1, child2
end

# MUTATION FUNCTION 

# This mutation function swaps two patients of two routes
function mutate4!(individual::Individual, instance::ProblemInstance)
    routes = individual.routes
    nb_nurses = length(routes)
    capacity = instance.capacity_nurse
    # Two random routes
    route1_idx = rand(1:nb_nurses)
    route2_idx = rand(setdiff(1:nb_nurses, [route1_idx]))
    if isempty(routes[route1_idx]) || isempty(routes[route2_idx])
        return
    end
    # Select two random patients
    patient1_idx = rand(1:length(routes[route1_idx]))
    patient2_idx = rand(1:length(routes[route2_idx]))
    # Check capacity first
    patient1 = routes[route1_idx][patient1_idx]
    patient2 = routes[route2_idx][patient2_idx]
    if (sum([instance.patients[string(patient)]["demand"] for patient in routes[route1_idx]]) - instance.patients[string(patient1)]["demand"] + instance.patients[string(patient2)]["demand"] > capacity) || (
        sum([instance.patients[string(patient)]["demand"] for patient in routes[route2_idx]]) - instance.patients[string(patient2)]["demand"] + instance.patients[string(patient1)]["demand"] > capacity
    )
        return
    end
    # Swap the two patients
    individual.routes[route1_idx][patient1_idx], 
    individual.routes[route2_idx][patient2_idx] = (
        individual.routes[route2_idx][patient2_idx],
        individual.routes[route1_idx][patient1_idx]
    )
end

# This mutation function shuffles a subset of a nurse's route
function mutate5!(individual::Individual, max_subset_size::Int=4)
    routes = individual.routes
    available_routes = [i for i in 1:length(routes) if !isempty(routes[i])]
    # random route
    route_idx = rand(available_routes)
    len_route = length(routes[route_idx])
    max_subset_size = min(max_subset_size, len_route)
    # shuffle a subset of the route
    elements_to_shuffle = rand(1:max_subset_size)
    shuffle_start_idx = rand(1:len_route-elements_to_shuffle+1)
    shuffle!(individual.routes[route_idx][shuffle_start_idx:shuffle_start_idx + elements_to_shuffle - 1])
end

# This mutation function is inserting a patient in someone else's route
function mutate6!(individual::Individual, instance::ProblemInstance)
    routes = individual.routes
    nb_nurses = length(routes)
    capacity = instance.capacity_nurse
    available_routes = [i for i in 1:nb_nurses if !isempty(routes[i])]
    route_to_remove_idx = rand(available_routes)
    # Select a random patient from route_to_remove
    patient_idx = rand(1:length(routes[route_to_remove_idx]))
    patient = routes[route_to_remove_idx][patient_idx]
    # Select a random route to insert the patient ; check if exists before removing
    route_to_insert_idx = rand(setdiff(available_routes, [route_to_remove_idx]))
    # check capacity
    if sum([instance.patients[string(patient)]["demand"] for patient in routes[route_to_insert_idx]]) + instance.patients[string(patient)]["demand"] > capacity
        println("no capacity")
        return
    end
    # Remove the patient from route_to_remove
    splice!(individual.routes[route_to_remove_idx], patient_idx)
    # Find the best insertion point ? flemme,l taer
    # Insert the patient in the new route
    insert!(individual.routes[route_to_insert_idx], rand(1:length(routes[route_to_insert_idx])), patient)
end

# This mutation splits a route into two, and puts half of the patients in a new route
function mutate7!(individual::Individual)
    routes = individual.routes
    nb_nurses = length(routes)
    empty_routes = [i for i in 1:nb_nurses if isempty(routes[i])]
    # Check if there are empty routes ; if not return because can't split
    isempty(empty_routes) && return
    # Take an empty route
    empty_route_to_fill_idx = rand(empty_routes)
    # Select one of the longest route and split it
    route_to_split_idx = argmax([length(route) for route in routes])
    route_to_split = routes[route_to_split_idx]
    # Split sequentially
    indexes_to_remove = [2*i for i in 1:div(length(route_to_split), 2)]
    for idx in reverse(indexes_to_remove)
        push!(routes[empty_route_to_fill_idx], splice!(route_to_split, idx))
    end
end


# Survival selection

function survivor_selection(population_fitnessed::Vector{Individual}, population_size::Int, tournament_size::Int=3, penalty_cost::Float64=10.0)::Vector{Individual}
    sort!(population_fitnessed, by = x -> x.score)
    elite_count = div(population_size, 10)
    elites = deepcopy(population_fitnessed[1:elite_count])
    # Tournament selection for the remaining population
    tournament_pop = []
    remaining_count = population_size - elite_count
    for i in 1:remaining_count
        selected_individuals = rand(population_fitnessed, tournament_size)
        winner = argmin(x -> x.score, selected_individuals)
        push!(tournament_pop, deepcopy(winner))
    end
    return vcat(elites, tournament_pop)
end

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
    #locker = ReentrantLock()
    # Main loop
    for generation in 1:num_generations
        next_population = deepcopy(population)
        if same_best_solution_counter % 5 == 4
            println("Same best solution for more than 5 generations, adding diversity")
            next_population = vcat(next_population, [random_solution(instance) for _ in 1:div(population_size, 2)], [position_cluster_solution(instance) for _ in 1:div(population_size, 2)])
        end
        for i in 1:(child_factor * population_size)
            # Select two parents
            parent1_idx, parent2_idx = tournament_selection(population, instance, fitness_cache, tournament_size, penalty_cost)
            parent1 = population[parent1_idx]
            parent2 = population[parent2_idx]

            # Crossover
            child1, child2 = crossover(parent1, parent2, instance, fitness_cache, penalty_cost)
            # Mutation
            
            mutation_rate > rand() ? mutate4!(child1, instance) : nothing
            mutation_rate > rand() ? mutate5!(child1) : nothing
            mutation_rate > rand() ? mutate6!(child1, instance) : nothing
            mutation_rate > rand() ? mutate7!(child1) : nothing
            mutation_rate > rand() ? mutate4!(child2, instance) : nothing
            mutation_rate > rand() ? mutate5!(child2) : nothing
            mutation_rate > rand() ? mutate6!(child2, instance) : nothing 
            mutation_rate > rand() ? mutate7!(child2) : nothing 
            
            # Add to offsprings
            #=begin
                lock(locker)
                try
                    push!(next_population, child1)
                    push!(next_population, child2)
                finally
                    unlock(locker)
                end
            end=#
            push!(next_population, child1)
            push!(next_population, child2)
        end
        println("End of crossover and mutations for generation $generation")

        # Survivor selection by elitism
        evaluations = [evaluate(sol, instance, fitness_cache, 1.0, max(penalty_cost/10, 0.4), penalty_cost, penalty_cost) for sol in next_population]
        feasabilities = [individual.feasability for individual in evaluations]
        feasability_rate = sum(feasabilities) / length(feasabilities)
        feasability_rate > 0 ? println("Feasibility POSITIVE!! \n") : println("Feasibility nulle \n")
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
        print("For generation $generation, best score is $(best_solution_score), travel time of $(best_solution_travel_time), average fitness is $(avg_fitness), feasability rate is $(feasability_rate), penalty cost is $(penalty_cost)\n")
    end
    # Return the best solution
    best_solution = argmin((x -> x.score), population)
    println("Best score found : $(best_solution)\n")
    return best_solution, population, fitnesses, scores, average_fitnesses
end
## MAIN ##

Random.seed!(1234)
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

best_solution91, population91, fitnesses91, scores91, average_fitnesses91 = genetic_algorithm(data_instance9, [1.0, 0, 0.0], 100, 300 , 3, 4, 0.1, 10.0)
best_solution92, population92, fitnesses92, scores92, average_fitnesses92 = genetic_algorithm(data_instance9, [0.0, 1.0, 0.0], 100, 300, 3, 4, 0.1, 10.0)
best_solution93, population93, fitnesses93, scores93, average_fitnesses93 = genetic_algorithm(data_instance9, [0.0, 0.0, 1.0], 100, 300, 3, 4, 0.1, 10.0)

best_solution90, population90, fitnesses90, scores90, average_fitnesses90 = genetic_algorithm(data_instance9, [0.5, 0.5, 0.0], 100, 900, 3, 4, 0.05, 10.0, true, [population91, population92, population93])


function plot_patient_positions(instance::ProblemInstance)
    x = [instance.patients[string(patient)]["x_coord"] for patient in 1:length(instance.patients)]
    y = [instance.patients[string(patient)]["y_coord"] for patient in 1:length(instance.patients)]
    labels = [string(patient) for patient in 1:length(instance.patients)]
    scatter(x, y, labels=labels, title="Patient positions", xlabel="x", ylabel="y")
end

function plot_routes(routes::Individual, instance::ProblemInstance)
    routes = routes.routes
    plot()
    x = [instance.patients[string(patient)]["x_coord"] for patient in 1:length(instance.patients)]
    y = [instance.patients[string(patient)]["y_coord"] for patient in 1:length(instance.patients)]
    scatter!(x, y, title="Patient positions")
    for (i, route) in enumerate(routes)
        # Patients coords
        x = [instance.patients[string(patient)]["x_coord"] for patient in route]
        y = [instance.patients[string(patient)]["y_coord"] for patient in route]
        # Depot coords
        depot_x = instance.depot["x_coord"]
        depot_y = instance.depot["y_coord"]
        # Add depot to routes
        pushfirst!(x, depot_x)
        pushfirst!(y, depot_y)
        push!(x, depot_x)
        push!(y, depot_y)
        plot!(x, y, ms=5, legend=false)
    end
    display(plot!(title="Routes"))
end

function output_solution(individual::Individual, instance::ProblemInstance, filename::String)
    open(filename, "w") do io
        # Écrire la capacité de l'infirmière et le temps de retour au dépôt
        println(io, "Nurse capacity : $(instance.capacity_nurse)")
        println(io, "Depot return time : $(instance.depot["return_time"])")
        println(io, "-"^100)

        # Parcourir chaque route de l'individu
        for (i, route) in enumerate(individual.routes)
            route_duration = 0.0
            nurse_demand = 0
            route_str = "D(0) -> "
            last_patient_id = 0
            elapsed_time = 0.0

            for patient_id in route
                patient = instance.patients[string(patient_id)]
                travel_time = get_travel_time(last_patient_id, patient_id, instance)
                arrival_time = elapsed_time + travel_time
                departure_time = arrival_time + patient["care_time"]
                time_window_start = patient["start_time"]
                time_window_end = patient["end_time"]
                route_str *= "$(patient_id)($(arrival_time) - $(departure_time)) [$(time_window_start) - $(time_window_end)] -> "
                route_duration += travel_time
                elapsed_time = departure_time
                nurse_demand += patient["demand"]
                last_patient_id = patient_id
            end

            # Ajouter le retour au dépôt
            travel_time = get_travel_time(last_patient_id, 0, instance)
            arrival_time = elapsed_time + travel_time
            route_str *= "D($(arrival_time))"
            route_duration += travel_time

            # Écrire les informations de la route
            println(io, "Nurse $(i) (N$(i))\t$(route_duration)\t$(nurse_demand)\t$(route_str)")
        end

        println(io, "-"^100)
        println(io, "Objective value (total duration): $(individual.total_travel_time)")
    end
end