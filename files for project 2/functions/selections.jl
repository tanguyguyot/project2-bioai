include("utilitaries.jl")
include("structures.jl")
include("evaluate.jl")

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