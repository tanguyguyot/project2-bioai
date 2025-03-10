include("structures.jl")

# Deprecated
function hash_individual(individual::Individual)::UInt64
    return hash(individual.routes)
end

# Testing function used to verify that everything goes well.
function check_unicity_of_routes(individual::Individual, instance::ProblemInstance)::Bool
    routes = vcat(individual.routes...)
    if sum(routes) != sum(1:length(instance.patients))
        println("problème d'unicité des routes : ", sum(routes), " != ", sum(1:length(instance.patients)))
        return false
    end
    return true
end

# Returns the travel time between two patients
function get_travel_time(i::Int, j::Int, instance::ProblemInstance)
    travel_time_matrix = instance.travel_time_matrix
    # adds +1 because of the depot which is 0
    return travel_time_matrix[i+1][j+1]
end

# Inserts a patient to the best neighbour according to evaluate function
function insert_to_best_neighbour!(patient::Int64, individual::Individual, instance::ProblemInstance, penalty_cost::Float64=350.0)
    routes = copy(individual.routes)
    available_routes = [i for i in 1:length(routes) if !isempty(routes[i])]
    lowest_score = Inf
    best_neighbour_idx = 0
    best_route_idx = 0
    for route_idx in available_routes
        for position in 1:length(routes[route_idx])+1
            candidate_route = copy(routes[route_idx])
            # Insert patient in the route
            insert!(candidate_route, position, patient)
            # check in hashmap : deprecated
            # individual_hash = hash_individual(individual)
            # score = haskey(fitness_cache, individual_hash) ? score = fitness_cache[individual_hash].score : evaluate(Individual([candidate_route], 0.0, 0.0, 0.0, false), instance, fitness_cache, 1.0, max(div(penalty_cost, 10), 1.0), penalty_cost, penalty_cost).score
            score = evaluate(Individual([candidate_route], 0.0, 0.0, 0.0, false), instance, 1.0, max(div(penalty_cost, 10), 1.0), penalty_cost, penalty_cost).score
            if score < lowest_score
                lowest_score = score
                best_neighbour_idx = position
                best_route_idx = route_idx
            end
        end
    end
    insert!(individual.routes[best_route_idx], best_neighbour_idx, patient)
end

# Inserts a patient to the best neighbours ; also checks if have capacity ; deprecated for now
function insert_to_closest_neighbour!(patient::Int64, individual::Individual, instance::ProblemInstance)
    routes = copy(individual.routes)
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

# output the whole population in a file
function output_population(population::Vector{Individual}, output_name::String="output.txt")
    open("./outputs/$output_name.txt", "w") do io
        println(io, population)
    end
end

# to log everything
function output_all_solution(sol, output_name)
    open("./outputs/solutions_$output_name.txt", "a") do io
        println(io, sol)
    end
end

# output a solution in the right format for the project delivery
function output_solution(individual::Individual, instance::ProblemInstance, filename::String)
    open("./outputs/$(filename)_formatted_solution.txt", "w") do io
        # Capacity and time to return to depot
        println(io, "Nurse capacity : $(instance.capacity_nurse)")
        println(io, "Depot return time : $(instance.depot["return_time"])")
        println(io, "-"^100)

        # For each individual route
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

            # Back to depot
            travel_time = get_travel_time(last_patient_id, 0, instance)
            arrival_time = elapsed_time + travel_time
            route_str *= "D($(arrival_time))"
            route_duration += travel_time

            # Route informations
            println(io, "Nurse $(i) (N$(i))\t$(route_duration)\t$(nurse_demand)\t$(route_str)")
        end

        println(io, "-"^100)
        println(io, "Objective value (total duration): $(individual.total_travel_time)")
    end
end