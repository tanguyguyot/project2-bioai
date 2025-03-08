# expected solution format : list of nurse route's lists i.e. each list is a different nurse's route

using JSON, Random, Plots, Clustering, BenchmarkTools
include("structures.jl")

# NB : peut-être utiliser des Sets ? vu qu'on ne visite qu'une seule fois chaque patient

train_data = JSON.parsefile("train/test_0.json")

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


# Function to save a solution to a json file

function save_solution_to_json(solution::Solution, instance::ProblemInstance, filename::String)
    routes = Dict(string(i) => route for (i, route) in enumerate(solution.routes))
    coordinates = Dict(string(patient) => (x = instance.patients[string(patient)]["x_coord"], y = instance.patients[string(patient)]["y_coord"]) for patient in 1:length(instance.patients))
    solution_dict = Dict("routes" => routes, "coordinates" => coordinates)
    open(filename, "w") do io
        JSON.print(io, solution_dict)
    end
end

# PRACTICAL FUNCTIONS

function get_travel_time(i::Int, j::Int)
    # adds +1 because of the depot which is 0
    return travel_time_matrix[i+1][j+1]
end

# Function that sort patients by start time
function get_sorted_patients(instance::ProblemInstance)
    patients = instance.patients
    sorted_patients = sort(collect(1:length(patients)), by = x -> (
        patients[string(x)]["start_time"],  # 1er critère : start_time
        x - patients[string(x)]["end_time"]  # 2e critère : end_time, ordre décroissant
    ))
    return sorted_patients
end

# Calculate the total travel time in a solution, ie the sum of travel times ; obsolète
#=
function total_travel_time(sol::Solution)
    travel_times::Vector{Float64} = []
    for (_, route) in enumerate(sol.routes)
        travel_time = 0.0
        current_node = 1
        for patient in route
            travel_time += get_travel_time(current_node, patient)
            current_node = patient
        end
        travel_time += get_travel_time(current_node, 1)
        push!(travel_times, travel_time)
    end
    return sum(travel_times)
end
=#


function insert_to_closest_neighbour!(patient::Int64, solution::Solution)
    routes = solution.routes
    lowest_distance = Inf
    best_neighbour = 0
    best_neighbour_route = 0
    best_neighbour_route_idx = 0
    
    for (idx, route) in enumerate(routes)
        for i in route
            distance = get_travel_time(patient, i)
            if distance < lowest_distance
                lowest_distance = distance
                best_neighbour = i
                best_neighbour_route = route
                best_neighbour_route_idx = idx
            end
        end
    end
    if (best_neighbour_route_idx == 0)
        println("ya une erreur")
        return
    end
    # Place between the best neighbour and the second closest that surrounds the best neighbour 
    best_neighbour_idx = findfirst(x -> x == best_neighbour, best_neighbour_route)
    if (best_neighbour_idx == 1 == length(best_neighbour_route)) # cas particulier de route à un seul patient
        insert!(solution.routes[best_neighbour_route_idx], 1, patient)
    elseif (best_neighbour_idx == 1) # c'est le début de la route mais il a un voisin d'après
        get_travel_time(0, patient) < get_travel_time(patient, best_neighbour_route[2]) ? insert!(solution.routes[best_neighbour_route_idx], 1, patient) : insert!(solution.routes[best_neighbour_route_idx], 2, patient)
    elseif (best_neighbour_idx == length(best_neighbour_route)) # c'est la fin de la route mais il a un voisin avant
        get_travel_time(patient, 0) < get_travel_time(best_neighbour_route[length(best_neighbour_route)], patient) ? insert!(solution.routes[best_neighbour_route_idx], length(best_neighbour_route)+1, patient) : insert!(solution.routes[best_neighbour_route_idx], length(best_neighbour_route)+1, patient)
    else
        patient_before = (best_neighbour_idx == 1 ? best_neighbour_route[1] : best_neighbour_route[best_neighbour_idx-1])
        patient_after = (best_neighbour_idx == length(best_neighbour_route) ? best_neighbour_route[best_neighbour_idx] : best_neighbour_route[best_neighbour_idx+1])
        get_travel_time(patient_before, patient) < get_travel_time(patient, patient_after) ? insert!(solution.routes[best_neighbour_route_idx], best_neighbour_idx, patient) : insert!(solution.routes[best_neighbour_route_idx], best_neighbour_idx+1, patient)
    end
end

function closest_neighbour(instance::ProblemInstance, patient::Int)::Int64
    lowest_distance = Inf
    best_neighbour = 0
    for i in 1:length(instance.patients)
        distance = get_travel_time(patient, i+1)
        if distance < lowest_distance
            lowest_distance = distance
            best_neighbour = i
        end
    end
    return best_neighbour
end

function feasible_rate(instance::ProblemInstance, fitness_list::Vector{Solution})
    feasible_routes = 0
    for solution in fitness_list
        if !(solution.penalties > 0)
            feasible_routes += 1
        end
    end
    return feasible_routes / length(fitness_list)
end

function sol2ind(solution::Solution)::Vector{Int64}
    routes = solution.routes
    return reduce(vcat, routes)
end

# This function takes an individual which is a single 100-length vector of integers into a solution (list of lists)
function ind2sol(individual::Vector{Int64}, instance::ProblemInstance)::Vector{Vector{Int64}}
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

# INITIALIZATION

# Cluster patients based on their position
function position_cluster_solution(instance::ProblemInstance)::Solution
    nb_patients = length(instance.patients)
    patients = collect(1:nb_patients)
    nb_nurses = instance.nbr_nurses
    coordinates = hcat([instance.patients[string(patient)]["x_coord"] for patient in 1:nb_patients], [instance.patients[string(patient)]["y_coord"] for patient in 1:nb_patients])'
    R = kmeans(coordinates, nb_nurses)
    clusters = R.assignments
    routes = [[] for _ in 1:nb_nurses]
    for patient in patients
        push!(routes[clusters[patient]], patient)
    end
    return Solution(routes, 0.0, 0.0, 0)   
end

function position_cluster_solution2(instance::ProblemInstance)::Vector{Int64}
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

function time_window_cluster_solution(instance::ProblemInstance)::Solution
    patients = instance.patients
    nb_patients = length(patients)
    nb_nurses = instance.nbr_nurses # nb de clusters pour le moment
    data = [ [(patients[string(patient)]["start_time"] + patients[string(patient)]["end_time"]) / 2, patients[string(patient)]["end_time"] - patients[string(patient)]["start_time"]] for patient in 1:nb_patients]
    data_matrix = hcat(data...)
    R = kmeans(data_matrix, nb_nurses)
    clusters = R.assignments
    routes = [[] for _ in 1:nb_nurses]
    clusters_in_routes = [[] for _ in 1:nb_nurses]
    nurses_load = zeros(nb_nurses)
    for (patient, cluster) in enumerate(clusters)
        available_nurses = findall(nurses_load .+ patients[string(patient)]["demand"] .<= instance.capacity_nurse)
        filter!(nurse -> !(cluster in clusters_in_routes[nurse]), available_nurses)
        nurse = rand(available_nurses)
        push!(routes[nurse], patient)
        push!(clusters_in_routes[nurse], cluster)
        nurses_load[nurse] += patients[string(patient)]["demand"]
    end
    for route in routes
        sort!(route, by = x -> instance.patients[string(x)]["start_time"])
    end
    return Solution(routes, 0.0, 0.0, 0)
end

function time_window_cluster_solution2(instance::ProblemInstance)::Vector{Int}
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
function time_window_solution_2(instance::ProblemInstance)::Solution
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

# Make a random solution, using greedy heuristic for respecting constraints more
function random_solution(instance::ProblemInstance)::Solution
    num_nurse::Int64 = instance.nbr_nurses
    # Initialize empty route list for each nurse
    routes::Vector{Vector{Int64}} = [[] for _ in 1:num_nurse]

    # Instead of shuffling patient ID, let's order them in order of start time : greedy heuristic
    patients::Vector{Int64} = get_sorted_patients(instance)

    nurse_loads::Vector{Int} = zeros(num_nurse)

    # Assign patients to nurses ; check if capacity is not exceeded
    for patient in patients
        demand::Int64 = instance.patients[string(patient)]["demand"]

        # Find valid nurses
        valid_nurses::Vector{Int64} = findall(nurse_loads .+ demand .<= instance.capacity_nurse)
        if isempty(valid_nurses)
            error("No valid nurse found for patient $patient! (Capacity exceeded)")
            return Solution([], Inf, Inf, Inf)
        end

        # Assign patient to a random nurse
        nurse = rand(valid_nurses)
        push!(routes[nurse], patient)
        nurse_loads[nurse] += demand
    end
    return Solution(routes, 0.0, 0.0, 0)
end

function random_solution2(instance::ProblemInstance)::Vector{Int}
    return randperm(length(instance.patients))
end

# Create a population of random solutions

function create_population(instance::ProblemInstance, population_size::Int)::Vector{Solution}
    dividence = div(population_size, 3)
    reste = population_size % 3
    return vcat([random_solution(instance) for _ in 1:(dividence + reste)], 
                [position_cluster_solution(instance) for _ in 1:dividence],
                [time_window_cluster_solution(instance) for _ in 1:dividence])
end

# EVALUATION FUNCTION

# Evaluate solution, applies penalties
function evaluate(solution::Solution, instance::ProblemInstance, penalty_coeff::Int64=10)::Solution
    travel_times::Vector{Float64} = []
    global_score = 0.0
    penalties = 0

    # Evaluate each nurse route
    for (nurse, route) in enumerate(solution.routes)
        current_node = 0 # 0 is the depot
        nurse_load = 0
        nurse_travel_time = 0.0
        nurse_care_time = 0.0
        nurse_total_time = 0.0

        for patient in route
            # patient time window values
            patient_start_time = instance.patients[string(patient)]["start_time"]
            patient_end_time = instance.patients[string(patient)]["end_time"]

            # total time spent by the nurse
            nurse_travel_time += get_travel_time(current_node, patient)
            nurse_care_time += instance.patients[string(patient)]["care_time"]

            # nurse load for capacity check
            nurse_load += instance.patients[string(patient)]["demand"]
            nurse_total_time = nurse_travel_time + nurse_care_time
            # Check if patient is visited within time window #
            if (patient_start_time > nurse_total_time)
                # if nurse arrives too early, wait
                nurse_care_time += patient_start_time - nurse_total_time
                nurse_total_time = nurse_travel_time + nurse_care_time
            end
            if (nurse_total_time > patient_end_time)
                # if nurse arrives too late, add penalty
                penalties += 1
            end
            current_node = patient
        end

        # Add travel time to the travel time list with return to depot time ; depot is 0
        nurse_travel_time += get_travel_time(current_node, 0)
        push!(travel_times, nurse_travel_time)

        # Capacity check for the nurse
        if nurse_load > instance.capacity_nurse
            penalties += 1
        end

        # Calculate the whole time spent by the nurse, and check if it exceeds the return time
        nurse_total_time = nurse_travel_time + nurse_care_time
        # adds penalty if return time exceeded
        penalties += max(0, 0.01 * (instance.depot["return_time"] - nurse_total_time))
    end
    total_travel_time = sum(travel_times)
    global_score = total_travel_time + penalty_coeff * penalties
    return Solution(solution.routes, total_travel_time, global_score, penalties)
end

# PARENT SELECTION

# A simple tournament selection function for parent selection

function tournament_selection(population::Vector{Solution}, instance::ProblemInstance, k::Int=3, penalty_coeff::Int64=10)::Tuple{Int64, Int64}
    idx_candidates1 = rand(1:length(population), k)
    candidates = [population[i] for i in idx_candidates1]
    min_idx_score1 = argmin([evaluate(x, instance, penalty_coeff).score for x in candidates])
    idx_candidates2 = rand(setdiff(1:length(population), min_idx_score1), k)
    candidates = [population[i] for i in idx_candidates2]
    min_idx_score2 = argmin([evaluate(x, instance, penalty_coeff).score for x in candidates])
    return idx_candidates1[min_idx_score1], idx_candidates2[min_idx_score2]
end

# CROSSOVER FUNCTION 

# Longest common subsequence function ; elite and worst
#=
function LCS(population::Vector{Solution}, instance::ProblemInstance, n::Int64)
    # Population fitnesses
    fitnesses = [evaluate(sol, instance).score for sol in population]
    # Sort population by fitness
    sorted_population = sort(population, by = x -> evaluate(x, instance).score)
    # Elite and worst solutions
    elite = sorted_population[1:n]
    worst = sorted_population[end-n:end]
end =#

# Crossover function like slides
function crossover(parent1::Solution, parent2::Solution, instance::ProblemInstance)
    # convert into list of routes to identify routes
    routes1::Vector{Vector{Int64}} = parent1.routes
    routes2::Vector{Vector{Int64}} = parent2.routes

    # initialize childs
    child1::Solution = deepcopy(parent1)
    child2::Solution = deepcopy(parent2)

    # Select two random nurses' routes from each parent
    route1_idx::Int64 = rand(1:length(routes1))
    route2_idx::Int64 = rand(1:length(routes2))
    route1::Vector{Int64} = routes1[route1_idx]
    route2::Vector{Int64} = routes2[route2_idx]

    # remove the patients of route1 from the first parent and vice versa
    for patient in route2
        # chercher le patient dans les routes de child1
        for route in routes1
            idx = findfirst(x -> x == patient, child1.routes)
            isnothing(idx) ? nothing : deleteat!(child1.routes, idx) 
        end
    end
    for patient in route1
        # chercher le patient dans les routes de child2
        for route in routes2
            idx = findfirst(x -> x == patient, child2.routes)
            isnothing(idx) ? nothing : deleteat!(child2.routes, idx)
        end
    end

    # now for each patient without a visitor, find the best insertion
    for patient in route2
        insert_to_closest_neighbour!(patient, child1)
    end
    for patient in route1
        insert_to_closest_neighbour!(patient, child2)
    end
    if (length(Set(child1.routes)) != length(child1.routes)) || (length(Set(child2.routes)) != length(child2.routes))
        println("Redundancy (crossover function)")
    end
    return child1, child2
end

# MUTATION FUNCTION 

# This mutation function swaps two patients in a nurse's route
function mutate(solution::Solution, mutation_rate::Float64=0.01)::Solution
    if rand() > mutation_rate
        return solution
    end
    nurses = solution.routes
    random_nurse = rand(1:length(nurses))
    while length(nurses[random_nurse]) == 0
        random_nurse = rand(1:length(nurses))
    end
    # Select two random patients
    patient1 = rand(nurses[random_nurse])
    patient2 = rand(nurses[random_nurse])
    if patient1 == patient2
        return solution
    end
    # Swap the two patients
    idx1 = findfirst(x -> x == patient1, nurses[random_nurse])
    idx2 = findfirst(x -> x == patient2, nurses[random_nurse])
    nurses[random_nurse][idx1], nurses[random_nurse][idx2] = solution.routes[random_nurse][idx2], solution.routes[random_nurse][idx1]
    return solution
end

# This mutation function shuffles a subset of a nurse's route
function mutate2(solution::Solution, mutation_rate::Float64=0.01)::Solution
    if rand() > mutation_rate
        return solution
    end
    nurses = solution.routes
    random_nurse = rand(1:length(nurses))
    # shuffle a subset of the nurse's route
    if length(nurses[random_nurse]) > 1
        idx1 = rand(1:length(nurses[random_nurse]))
        idx2 = rand(1:length(nurses[random_nurse]))
        idx1, idx2 = min(idx1, idx2), max(idx1, idx2)
        shuffle!(nurses[random_nurse][idx1:idx2])
    end
    return solution
end

# This mutation function is inserting a patient in someone else's route
function mutate3(solution::Solution, instance::ProblemInstance, mutation_rate::Float64=0.01)::Solution
    if rand() > mutation_rate
        return solution
    end
    nurses = solution.routes
    random_nurse1 = rand(1:length(nurses))
    random_nurse2 = rand(1:length(nurses))
    while random_nurse1 == random_nurse2
        random_nurse2 = rand(1:length(nurses))
    end
    # Select a random patient from the first nurse
    if length(nurses[random_nurse1]) > 0
        demand = instance.patients[string(rand(nurses[random_nurse1]))]["demand"]
        # Check if the second nurse can handle the patient
        if length(nurses[random_nurse2]) == 0
            return solution
        end
        if sum([instance.patients[string(patient)]["demand"] for patient in nurses[random_nurse2]]) + demand > instance.capacity_nurse
            return solution
        end
        patient = rand(nurses[random_nurse1])
        # Insert the patient in the second nurse's route
        push!(nurses[random_nurse2], patient)
        deleteat!(nurses[random_nurse1], findfirst(x -> x == patient, nurses[random_nurse1]))
    end
    return solution
end

# Survival selection

# Fonction d'élitisme pour la sélection des survivants
function elitism_selection(population::Vector{Solution}, elite_size::Int)::Vector{Solution}
    sorted_population = sort(population, by = x -> x.score)
    return sorted_population[1:elite_size]
end

# GENETIC ALGORITHM 

# Genetic algorithm 2 functions
function genetic_algorithm2(instance::ProblemInstance, penalty_coeff::Int64=10, num_generations::Int=500, population_size::Int=500, mutation_rate::Float64=0.01, tournament_amount::Int64=3, offspring_factor::Int64=3)
    # Initialize population
    population = vcat([random_solution(instance) for _ in 1:div(population_size, 2)], [position_cluster_solution(instance) for _ in 1:div(population_size, 2)])
    # Evaluate population
    population = [evaluate(sol, instance, penalty_coeff) for sol in population]

    # Keep track of scores and travel travel_times
    travel_times = []
    scores = []
    penalties = []
    feasabilities = []
    penalty_factor = 1

    # Main loop
    for generation in 1:num_generations
        # Create next generation initialized with the current population
        next_population = population
        for i in 1:offspring_factor * population_size
            # crossover
            parent1_idx, parent2_idx = tournament_selection(population, instance, tournament_amount, penalty_coeff)
            child1, child2 = crossover(population[parent1_idx], population[parent2_idx], instance)

            # mutation : mutation_rate of 0.01 means that 1% of the time, a mutation will occur (1 for 100 individuals)
            child1 = mutate(child1, mutation_rate)
            child2 = mutate(child2, mutation_rate)
            child1 = mutate2(child1, mutation_rate)
            child2 = mutate2(child2, mutation_rate)
            child1 = mutate3(child1, instance, mutation_rate)
            child2 = mutate3(child2, instance, mutation_rate)

            push!(next_population, child1)
            push!(next_population, child2)
        end
        
        # Survivor selection by elitism
        fitnesses::Vector{Solution} = [evaluate(sol, instance, penalty_coeff) for sol in next_population]
        population = elitism_selection(fitnesses, population_size)
        feasible_solutions_rate = feasible_rate(instance, population)
        penalty_factor = (feasible_solutions_rate < 0.1 ? 2 : 1)
        
        best_solution = argmin((x -> x.score), fitnesses)
        push!(travel_times, best_solution.total_travel_time)
        push!(scores, best_solution.score)
        push!(penalties, best_solution.penalties)
        push!(feasabilities, feasible_solutions_rate)
        print("For generation $generation, best score is $(best_solution.score), penalties are $(best_solution.penalties), distance is $(best_solution.total_travel_time), feasibility is $(feasible_solutions_rate)\n")
    end

    # Return the best solution
    fitnesses = [evaluate(sol, instance, penalty_coeff) for sol in population]
    best_solution = argmin((x -> x.score), fitnesses)
    println("Best solution found : $(best_solution.score)\n")
    println("Travel time : $(best_solution.total_travel_time)\n")
    return best_solution, fitnesses, travel_times, scores, penalties, feasabilities
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

function plot_clusters(instance::ProblemInstance, solution::Solution)
    for (i, route) in enumerate(solution.routes)
        x = [instance.patients[string(patient)]["x_coord"] for patient in route]
        y = [instance.patients[string(patient)]["y_coord"] for patient in route]
        scatter!(x, y, xlabel="x", ylabel="y")
    end
    display(scatter!())
end

function plot_clusters_time_windows(instance::ProblemInstance, solution::Solution)
    for (i, route) in enumerate(solution.routes)
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

function plot_routes(solution::Solution, instance::ProblemInstance)
    plot()
    x = [instance.patients[string(patient)]["x_coord"] for patient in 1:length(instance.patients)]
    y = [instance.patients[string(patient)]["y_coord"] for patient in 1:length(instance.patients)]
    labels = [string(patient) for patient in 1:length(instance.patients)]
    scatter!(x, y, labels=labels, title="Patient positions", xlabel="x", ylabel="y")
    for (i, route) in enumerate(solution.routes)
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