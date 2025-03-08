
include("utilitaries.jl")
include("structures.jl")

# This mutation function swaps two patients of two routes
function mutate4!(individual::Individual, instance::ProblemInstance)
    routes = copy(individual.routes)
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
    routes = copy(individual.routes)
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
    routes = copy(individual.routes)
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
    routes = copy(individual.routes)
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
        push!(individual.routes[empty_route_to_fill_idx], splice!(route_to_split, idx))
    end
end

# This mutate will merge two routes
function mutate8!(individual::Individual)
    routes = copy(individual.routes)
    nb_nurses = length(routes)
    available_routes = [i for i in 1:nb_nurses if !isempty(routes[i])]
    # Select two random routes
    route1_idx = rand(available_routes)
    route2_idx = rand(setdiff(available_routes, [route1_idx]))
    # Merge the two routes sequentially
    merged_route = []
    length_difference = abs(length(routes[route1_idx]) - length(routes[route2_idx]))
    min_length = min(length(routes[route1_idx]), length(routes[route2_idx]))
    for i in 1:min_length
        push!(merged_route, routes[route1_idx][i])
        push!(merged_route, routes[route2_idx][i])
    end
    if length_difference > 0
        if length(routes[route1_idx]) > length(routes[route2_idx])
            append!(merged_route, routes[route1_idx][min_length+1:end])
        else
            append!(merged_route, routes[route2_idx][min_length+1:end])
        end
    end
    # check we got all the patients
    if length(merged_route) != length(routes[route1_idx]) + length(routes[route2_idx])
        println("Error in merging routes")
        return
    end
    individual.routes[route1_idx] = merged_route
    individual.routes[route2_idx] = []
end

# This function splits a route in half, not sequentially
function mutate9!(individual::Individual)
    routes = copy(individual.routes)
    nb_nurses = length(routes)
    empty_routes = [i for i in 1:nb_nurses if isempty(routes[i])]
    # if no empty route, return because not useful
    isempty(empty_routes) && return
    # Select an empty route id to fill in
    empty_route_idx = empty_routes[1]
    # Select one of the longest routes
    route_to_split_idx = argmax([length(route) for route in routes])
    route = routes[route_to_split_idx]
    # Split the route in half
    split_idx = div(length(route), 2)
    individual.routes[route_to_split_idx] = route[1:split_idx]
    individual.routes[empty_route_idx] = route[split_idx+1:end]
end

# This mutation will merge two routes
function mutate10!(individual::Individual)
    routes = copy(individual.routes)
    nb_nurses = length(routes)
    available_routes = [i for i in 1:nb_nurses if !isempty(routes[i])]
    # Select two random routes
    route1_idx = rand(available_routes)
    route2_idx = rand(setdiff(available_routes, [route1_idx]))
    # Merge the two routes
    merged_route = vcat(routes[route1_idx], routes[route2_idx])
    individual.routes[route1_idx] = merged_route
    individual.routes[route2_idx] = []
end

