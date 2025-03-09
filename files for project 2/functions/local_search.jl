include("structures.jl")
include("evaluate.jl")

# quick search on the best individual to set small changes that can slightly improve solution
function local_search(individual::Individual, instance::ProblemInstance, penalty_cost::Float64=350.0)::Individual
    routes = copy(individual.routes)
    unempty_routes = [i for i in 1:length(routes) if !isempty(routes[i])]
    min_score = evaluate(individual, instance, 1.0, max(div(penalty_cost, 10), 1.0), penalty_cost, penalty_cost).score
    new_individual = deepcopy(individual)
    for idx in unempty_routes
        for i in 1:length(routes[idx])-1
            candidate_route = copy(routes[idx])
            candidate_individual = deepcopy(individual)
            candidate_route[i], candidate_route[i+1] = candidate_route[i+1], candidate_route[i]
            candidate_individual.routes[idx] = candidate_route
            candidate_individual = evaluate(candidate_individual, instance, 1.0, max(div(penalty_cost, 10), 1.0), penalty_cost, penalty_cost)
            if candidate_individual.score < min_score
                println("local search occurred")
                min_score = candidate_individual.score
                new_individual.routes[idx] = candidate_route
                break
            end
        end
    end
    return evaluate(new_individual, instance, 1.0, max(div(penalty_cost, 10), 1.0), penalty_cost, penalty_cost)
end