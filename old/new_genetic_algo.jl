# expected solution format : list of nurse route's lists i.e. each list is a different nurse's route

using JSON, Random, Plots, Clustering, BenchmarkTools
include("structures.jl")

# NB : peut-être utiliser des Sets ? vu qu'on ne visite qu'une seule fois chaque patient

train_data = JSON.parsefile("train/train_9.json")

data_instance = ProblemInstance(
    train_data["depot"],
    train_data["patients"],
    train_data["nbr_nurses"],
    train_data["instance_name"],
    train_data["travel_times"],
    train_data["benchmark"],
    train_data["capacity_nurse"]
)

travel_time_matrix = train_data["travel_times"]

# PRACTICAL FUNCTIONS

function get_travel_time(i::Int, j::Int)
    # adds +1 because of the depot which is 0
    return travel_time_matrix[i+1][j+1]
end

function insert_to_closest_neighbour!(patient::Int64, individual::Vector{Int64})
    lowest_distance = Inf
    best_neighbour = 0
    for i in individual
        distance = get_travel_time(patient, i)
        if distance < lowest_distance
            lowest_distance = distance
            best_neighbour = i
        end
    end
    # Place between the best neighbour and the second closest that surrounds the best neighbour 
    best_neighbour_idx = findfirst(x -> x == best_neighbour, individual)
    patient_before = (best_neighbour_idx == 1 ? individual[1] : individual[best_neighbour_idx-1])
    patient_after = (best_neighbour_idx == length(individual) ? individual[best_neighbour_idx] : individual[best_neighbour_idx+1])
    get_travel_time(patient_before, patient) < get_travel_time(patient, patient_after) ? insert!(individual, best_neighbour_idx, patient) : insert!(individual, best_neighbour_idx+1, patient)
end

function sol2ind(solution::Solution)::Vector{Int64}
    routes = solution.routes
    return reduce(vcat, routes)
end

# This function takes an individual which is a single 100-length vector of integers into a solution (list of lists)
function ind2sol(individual::Vector{Int64}, instance::ProblemInstance, max_ratio::Float64=1.0)::Vector{Vector{Int64}}
    routes = []
    nurse_capacity = instance.capacity_nurse
    nurse_due_time = instance.depot["return_time"]
    # Initialize a route
    nurse_route = []
    nurse_load = 0
    elapsed_time::Float64 = 0.0
    last_patient_id = 0
    for patient_id in individual
        # Update vehicle load
        demand = instance.patients[string(patient_id)]["demand"]
        updated_nurse_load = nurse_load + demand
        # Update elapsed time
        care_time = instance.patients[string(patient_id)]["care_time"]
        return_time = get_travel_time(patient_id, 0) 
        updated_elapsed_time = elapsed_time + get_travel_time(last_patient_id, patient_id) + care_time + return_time
        # Validate nurse load and elapsed time
        if (updated_nurse_load <= nurse_capacity) && (updated_elapsed_time <= nurse_due_time)
            # Add to current nurse route
            push!(nurse_route, patient_id)
            # Update nurse load
            nurse_load = updated_nurse_load
            # Update elapsed time
            elapsed_time = updated_elapsed_time
        else
            # Save current nurse route
            push!(routes, nurse_route)
            # Initialize a new route
            nurse_route = [patient_id]
            # Update nurse load
            nurse_load = demand
            # Update elapsed time
            elapsed_time = get_travel_time(0, patient_id) + care_time + return_time
        end
        # Update last patient ID
        last_patient_id = patient_id
    end
    if !isempty(nurse_route)
        # Save current nurse route before return if not empty
        push!(routes, nurse_route)
    end
    return routes
end

function find_max_ratio(nb_nurses)::Float64
    max_ratio = 1.0
    population = create_population(data_instance, 1000)
    individuals_to_solution = [ind2sol(individual, data_instance, 1.0) for individual in population]
    median_amount_of_routes = length(individuals_to_solution[div(length(individuals_to_solution), 2)])
    sens = sign(nb_nurses - median_amount_of_routes)
    max_ratio -= sens * 0.05
    while median_amount_of_routes != nb_nurses
        population = create_population(data_instance, 1000)
        individuals_to_solution = [ind2sol(individual, data_instance, max_ratio) for individual in population]
        median_amount_of_routes = length(individuals_to_solution[div(length(individuals_to_solution), 2)])
        max_ratio -= sens * 0.05
        println("max ratio is $max_ratio and median amount of routes is $median_amount_of_routes") 
    end
    return max_ratio
end

# INITIALIZATION

# Cluster patients based on their position

function position_cluster_solution(instance::ProblemInstance)::Vector{Int64}
    nb_patients = length(instance.patients)
    nb_nurses = instance.nbr_nurses
    coordinates = hcat([instance.patients[string(patient)]["x_coord"] for patient in 1:nb_patients], [instance.patients[string(patient)]["y_coord"] for patient in 1:nb_patients])'
    R = kmeans(coordinates, nb_nurses)
    clusters = R.assignments
    routes = []
    for cluster in 1:nb_nurses
        patients = findall(x -> x == cluster, clusters)
        append!(routes, patients)
    end
    return routes 
end

# Cluster patients based on their time window

function time_window_cluster_solution(instance::ProblemInstance)::Vector{Int}
    patients = instance.patients
    nb_patients = length(patients)
    nb_nurses = instance.nbr_nurses # nb de clusters pour le moment
    data = [ [(patients[string(patient)]["start_time"] + patients[string(patient)]["end_time"]) / 2, patients[string(patient)]["end_time"] - patients[string(patient)]["start_time"]] for patient in 1:nb_patients]
    data_matrix = hcat(data...)
    R = kmeans(data_matrix, nb_nurses)
    clusters = R.assignments
    routes = []
    cluster_count = 1
    while sum(clusters) > 0
        patient = findfirst(x -> x == cluster_count, clusters)
        if isnothing(patient)
            cluster_count > nb_nurses ? cluster_count = 1 : cluster_count += 1
            continue
        end
        append!(routes, patient)
        cluster_count > nb_nurses ? cluster_count = 1 : cluster_count += 1
        clusters[patient] = 0
    end
    return routes
end

# solution non aléatoire
function time_window_solution(instance::ProblemInstance)::Solution
    num_nurse::Int64 = instance.nbr_nurses
    num_patients::Int64 = length(instance.patients)
    capacity::Int64 = instance.capacity_nurse
    # Initialize empty route list for each nurse
    routes::Vector{Vector{Int64}} = [[] for _ in 1:num_nurse]
    sorted_patients = get_sorted_patients(instance)
    nurse_loads::Vector{Int} = zeros(num_nurse)
    nurse_number = 1
    while length(sorted_patients) > 0
        patient = popfirst!(sorted_patients)
        demand::Int64 = instance.patients[string(patient)]["demand"]
        if nurse_number > num_nurse
            nurse_number = 1
        end
        if nurse_loads[nurse_number] + demand > capacity
            nurse_number += 1
        end
        if nurse_number > num_nurse
            nurse_number = 1
        end
        push!(routes[nurse_number], patient)
        nurse_loads[nurse_number] += demand
        nurse_number += 1
    end 
    return Solution(routes, 0.0, 0.0, 0)
end

# Completely random solution
function random_solution(instance::ProblemInstance)::Vector{Int}
    return randperm(length(instance.patients))
end

# Create a population of solutions, based on the initialization functions

function create_population(instance::ProblemInstance, population_size::Int)::Vector{Vector{Int64}}
    dividende = div(population_size, 3)
    reste = population_size % 3
    return vcat([random_solution(instance) for _ in 1:(dividende + reste)], 
                [position_cluster_solution(instance) for _ in 1:dividende],
                [time_window_cluster_solution(instance) for _ in 1:dividende])
end

# EVALUATION FUNCTION

function evaluate(routes::Vector{Vector{Int64}}, instance::ProblemInstance, unit_cost::Float64=1.0, wait_cost=1.0, delay_cost=1.0)
    total_cost = 0
    # Find the best route splitting
    feasability = true
    for route in routes
        # the nurse's route time cost which is the non-respect of time window cost
        route_time_cost = 0
        route_distance = 0
        elapsed_time = 0
        last_patient_id = 0
        for patient_id in route
            # Calculate section distance
            distance = get_travel_time(last_patient_id, patient_id)
            # Update route_distance
            route_distance += distance
            # Calculate time cost
            arrival_time = elapsed_time + distance
            wait_time = max(instance.patients[string(patient_id)]["start_time"] - arrival_time, 0)
            late_time = max(arrival_time - instance.patients[string(patient_id)]["end_time"], 0)
            if late_time > 0
                feasability = false
            end
            time_window_cost = wait_cost * wait_time + delay_cost * late_time
            # Update nurse time cost
            route_time_cost += time_window_cost
            # Update elapsed time
            elapsed_time = arrival_time + instance.patients[string(patient_id)]["care_time"] + wait_time
            # Update last patient ID
            last_patient_id = patient_id
        end
        # Calculate back to depot cost
        route_distance += get_travel_time(last_patient_id, 0)
        # Calculate route cost
        route_cost = unit_cost * route_distance + route_time_cost
        # Update total cost
        total_cost += route_cost
    end
    return total_cost, feasability
end

# PARENT SELECTION

# A simple tournament selection function for parent selection

function tournament_selection(population::Vector{Vector{Int}}, instance::ProblemInstance, k::Int=3, max_ratio=1.0)::Int64
    idx_candidates = rand(1:length(population), k)
    candidates = [population[i] for i in idx_candidates]
    sol_candidates = [ind2sol(candidate, instance, max_ratio) for candidate in candidates]
    min_idx_score = argmin([evaluate(x, instance)[1] for x in sol_candidates])
    return idx_candidates[min_idx_score]
end

# CROSSOVER FUNCTION 

# Crossover function, partially matched crossover
function crossover(parent1::Vector{Int64}, parent2::Vector{Int64}, instance::ProblemInstance)
    # convert into list of routes to identify routes
    routes1 = ind2sol(parent1, instance)
    routes2 = ind2sol(parent2, instance)

    # initialize childs
    child1::Vector{Int64} = deepcopy(parent1)
    child2::Vector{Int64} = deepcopy(parent2)

    # Select two random nurses' routes from each parent
    route1_idx::Int64 = rand(1:length(routes1))
    route2_idx::Int64 = rand(1:length(routes2))
    route1 = routes1[route1_idx]
    route2 = routes2[route2_idx]

    # remove the patients of route1 from the first parent and vice versa
    for patient in route2
        idx = findfirst(x -> x == patient, child1)
        isnothing(idx) ? println("error, empty idx for route2") : deleteat!(child1, idx) 
    end
    for patient in route1
        idx = findfirst(x -> x == patient, child2)
        isnothing(idx) ? println("error, empty idx for route1") : deleteat!(child2, idx)
    end

    # now for each patient without a visitor, find the best insertion
    for patient in route2
        insert_to_closest_neighbour!(patient, child1)
    end
    for patient in route1
        insert_to_closest_neighbour!(patient, child2)
    end
    if (length(Set(child1)) != length(child1)) || (length(Set(child2)) != length(child2))
        println("Redundancy (crossover function)")
    end
    return child1, child2
end

# MUTATION FUNCTION 

# This mutation function swaps two patients in a nurse's route
function mutate4!(individual::Vector{Int})
    nb_patients = length(individual)
    # Select two random patients
    patient1 = rand(1:nb_patients)
    patient2 = rand(setdiff(1:nb_patients, [patient1]))
    # Swap the two patients
    individual[patient1], individual[patient2] = individual[patient2], individual[patient1]
end

# This mutation function shuffles a subset of a nurse's route
function mutate5!(individual::Vector{Int}, max_subset_size::Int=5)
    nb_patients = length(individual)
    # shuffle a subset of the route
    elements_to_shuffle = rand(2:max_subset_size)
    idx = rand(1:(nb_patients - elements_to_shuffle))
    shuffle!(individual[idx:idx+elements_to_shuffle])
end

# This mutation function is inserting a patient in someone else's route
function mutate6!(individual::Vector{Int})
    nb_patients = length(individual)
    # Select a random patient
    patient_idx = rand(1:nb_patients)
    # Moves the patient somewhere else in the individual list
    new_idx = rand(1:nb_patients)
    insert!(individual, new_idx, splice!(individual, patient_idx))
end

# Survival selection

# GENETIC ALGORITHM 

function genetic_algorithm(instance::ProblemInstance, num_generations::Int=500, population_size::Int=500, tournament_size::Int=3, child_factor::Int=3, mutation_rate::Float64=0.01)
    # Initialize a population
    population = create_population(instance, population_size)
    nb_nurses = instance.nbr_nurses

    # Keep track of scores and travel times
    scores = []
    feasabilities_record = []
    average_fitnesses = []
    fitnesses = []

    # set max_ratio
    max_ratio = 0.4

    individuals_to_solution::Vector{Vector{Vector{Int64}}} = [ind2sol(individual, instance, max_ratio) for individual in population]
    feasabilities = [evaluate(sol, instance)[2] for sol in individuals_to_solution]
    feasability_rate = sum(feasabilities) / length(feasabilities)

    # Main loop
    for generation in 1:num_generations
        next_population = population
        for i in 1:(child_factor * population_size)
            # Select two parents
            parent1 = population[tournament_selection(population, instance, tournament_size, max_ratio)]
            parent2 = population[tournament_selection(population, instance, tournament_size, max_ratio)]
            # Crossover
            child1, child2 = crossover(parent1, parent2, instance)
            # Mutation
            mutation_rate > rand() ? mutate4!(child1) : nothing
            mutation_rate > rand() ? mutate5!(child1) : nothing
            mutation_rate > rand() ? mutate6!(child1) : nothing
            mutation_rate > rand() ? mutate4!(child2) : nothing
            mutation_rate > rand() ? mutate5!(child2) : nothing
            mutation_rate > rand() ? mutate6!(child2) : nothing 
            # Add to offsprings
            push!(next_population, child1)
            push!(next_population, child2)
        end
        println("End of crossover and mutations for generation $generation")
        # Survivor selection by elitism
        # Si toutes les solutions sont pas faisables on double les pénaltys
        individuals_to_solution = [ind2sol(individual, instance, max_ratio) for individual in next_population]
        if feasability_rate == 0.0
            println("Feasability toujours nulle")
            evaluations = [evaluate(sol, instance, 1.0, 2.0, 2.0) for sol in individuals_to_solution]
        end
        evaluations = [evaluate(sol, instance) for sol in individuals_to_solution]

        # change max_ratio if not enough vehicles or too many vehicles
        median_amount_of_routes = length(individuals_to_solution[div(length(individuals_to_solution), 2)])
        
        println("nb de vehicules : $median_amount_of_routes")
        
        fitnesses = [evaluation[1] for evaluation in evaluations]
        fitness_population = [(idx, fitness) for (idx, fitness) in enumerate(fitnesses)]
        sort!(fitness_population, by = x -> x[2]) # fitness_population is sorted by fitness
        reduced_fitness = [x for x in fitness_population[1:population_size]]
        population = [next_population[x[1]] for x in reduced_fitness]

        # Feasability calculations
        feasabilities = [evaluation[2] for evaluation in evaluations]
        reduced_feasabilities = [feasabilities[x[1]] for x in reduced_fitness]
        feasability_rate = sum(reduced_feasabilities) / length(reduced_feasabilities)
        append!(feasabilities_record, feasability_rate)

        # Keep track of scores and average fitnesses
        best_solution = argmin((x -> x), fitnesses)
        push!(scores, best_solution)
        avg_fitness = sum(fitnesses) / length(fitnesses)
        push!(average_fitnesses, avg_fitness)
        print("For generation $generation, best score is $(best_solution), average fitness is $(avg_fitness), feasability rate is $(feasability_rate)\n")
    end
    # Return the best solution
    best_score = argmin((x -> x), fitnesses)
    best_solution_idx = findfirst(x -> x == best_score, fitnesses)
    best_solution = population[best_solution_idx]
    println("Best score found : $(best_solution)\n")
    return best_solution, best_solution_idx, population, fitnesses, scores, average_fitnesses
end
## MAIN ##

# testing the algo : MAIN

sol0 = random_solution(data_instance)
# genetic_algorithm3(data_instance, 10, 300)

#=
best_solution, final_population, travel_times, scores, penalties, feasabilities = genetic_algorithm(data_instance, 10, 50, 100, 0.01, 10, 5)

# Plot solution to check convergence speed
display(plot(travel_times, title="Travel times", label="Travel times", xlabel="Generation", ylabel="Travel time"))
display(plot(scores, title="Scores", label="Scores", xlabel="Generation", ylabel="Score"))
display(plot(penalties, title="Penalties", label="Penalties", xlabel="Generation", ylabel="Penalties"))


# To create a completely random solution, used for benchmarking
 function totally_random_solution(instance::ProblemInstance)::Solution
    num_nurse = instance.nbr_nurses
    # Initialize empty route list for each nurse
    routes = [[] for _ in 1:num_nurse]

    # Shuffle patient ID
    patients = shuffle(collect(1:length(instance.patients)))

    # Assign patients to nurses
    for (i, patient) in enumerate(patients)
        nurse = rand(1:num_nurse)
        push!(routes[nurse], patient)
    end
    return Solution(routes, 0.0, 0.0, 0)
end =#

# Plot functions

function plot_patient_positions(instance::ProblemInstance)
    x = [instance.patients[string(patient)]["x_coord"] for patient in 1:length(instance.patients)]
    y = [instance.patients[string(patient)]["y_coord"] for patient in 1:length(instance.patients)]
    labels = [string(patient) for patient in 1:length(instance.patients)]
    scatter(x, y, labels=labels, title="Patient positions", xlabel="x", ylabel="y")
end

function plot_clusters(instance::ProblemInstance, solution::Vector{Int})
    routes = ind2sol(solution, instance)
    plot()
    for (i, route) in enumerate(routes)
        x = [instance.patients[string(patient)]["x_coord"] for patient in route]
        y = [instance.patients[string(patient)]["y_coord"] for patient in route]
        scatter!(x, y, xlabel="x", ylabel="y")
    end
    display(scatter!())
end

function plot_clusters_time_windows(instance::ProblemInstance, solution::Vector{Int})
    routes = ind2sol(solution, instance)
    plot()
    for (i, route) in enumerate(routes)
        x = [(instance.patients[string(patient)]["start_time"] + instance.patients[string(patient)]["end_time"]) / 2 for patient in route]
        y = [instance.patients[string(patient)]["end_time"] - instance.patients[string(patient)]["start_time"] for patient in route]
        scatter!(x, y, xlabel="Median time", ylabel="Time window")
    end
    display(scatter!())
end

function plot_time_windows(instance::ProblemInstance)
    patients = instance.patients
    x = [patients[string(patient)]["start_time"] for patient in 1:length(patients)]
    # x2 : médiane start_time et end_time de chaque patient
    x2 = [(patients[string(patient)]["start_time"] + patients[string(patient)]["end_time"]) / 2 for patient in 1:length(patients)]
    y = [patient for patient in 1:length(patients)]
    scatter(x, y, xlabel="Start time", ylabel="Patient ID")
    scatter!(x2, y, xlabel="Median time", ylabel="Patient ID")
    display(scatter!())
end

function plot_routes(routes::Vector{Vector{Int}}, instance::ProblemInstance)
    plot()
    x = [instance.patients[string(patient)]["x_coord"] for patient in 1:length(instance.patients)]
    y = [instance.patients[string(patient)]["y_coord"] for patient in 1:length(instance.patients)]
    labels = [string(patient) for patient in 1:length(instance.patients)]
    scatter!(x, y, labels=labels, title="Patient positions", xlabel="x", ylabel="y")
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
        plot!(x, y, xlabel="x", ylabel="y", ms=5)
    end
    display(plot!(title="Routes"))
end