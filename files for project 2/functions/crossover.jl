include("utilitaries.jl")
include("structures.jl")
include("evaluate.jl")

function crossover(parent1::Individual, parent2::Individual, instance::ProblemInstance, fitness_cache::Dict{UInt64, Individual}, locker::ReentrantLock, penalty_cost::Float64=350.0)::Tuple{Individual, Individual}
    # convert into list of routes to identify routes
    routes1 = copy(parent1.routes)
    routes2 = copy(parent2.routes)
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
        insert_to_best_neighbour!(patient, child1, instance, fitness_cache, locker, penalty_cost)
    end
    for patient in route1
        insert_to_best_neighbour!(patient, child2, instance, fitness_cache, locker, penalty_cost)
    end
    return child1, child2
end