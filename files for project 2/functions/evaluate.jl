include("utilitaries.jl")
include("structures.jl")

function evaluate(individual::Individual, instance::ProblemInstance, fitness_cache::Dict{UInt64, Individual}, locker::ReentrantLock, unit_cost::Float64=1.0, delay_cost=2.0, capacity_cost=50.0, late_depot_cost=50.0,)::Individual
    individual_hash = hash_individual(individual)
    lock(locker) do
        if haskey(fitness_cache, individual_hash)
            return fitness_cache[individual_hash]
        end
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
    lock(locker) do
        fitness_cache[individual_hash] = evaluated_individual
    end
    return evaluated_individual
end