include("genetic_algo2.jl")
include("plotting.jl")

const NICHE_WEIGHTS = [[0.3, 0.7, 0], [0.7, 0.3, 0], [0.0, 0.0, 1.0]]
const MERGE_WEIGHT = [0.5, 0.5, 0]
const NICHE_PARAMS = (num_generations=60, population_size=100, tournament_size=3, child_factor=3, mutation_rate=0.15, penalty_rate=300.0)
const MERGE_PARAMS = (num_generations=30, population_size=300, tournament_size=3, child_factor=3, mutation_rate=0.15, penalty_rate=500.0)

function run_niche_algorithms(instance::ProblemInstance, output_prefix::String)
    solutions = []
    for (i, weight) in enumerate(NICHE_WEIGHTS)
        println("Data instance, niche $i")
        sol = genetic_algorithm(
            instance, weight,
            NICHE_PARAMS.num_generations,
            NICHE_PARAMS.population_size,
            NICHE_PARAMS.tournament_size,
            NICHE_PARAMS.child_factor,
            NICHE_PARAMS.mutation_rate,
            NICHE_PARAMS.penalty_rate,
            "$(output_prefix)_$i"
        )
        push!(solutions, sol)
    end
    return solutions
end

function run_merge_algorithm(instance::ProblemInstance, output_prefix::String, populations::Vector)
    println("Island merge")
    return genetic_algorithm(
        instance, MERGE_WEIGHT,
        MERGE_PARAMS.num_generations,
        MERGE_PARAMS.population_size,
        MERGE_PARAMS.tournament_size,
        MERGE_PARAMS.child_factor,
        MERGE_PARAMS.mutation_rate,
        MERGE_PARAMS.penalty_rate,
        output_prefix,
        true,
        populations
    )
end

function output_all_solutions(niche_solutions::Vector, merge_solution, instance::ProblemInstance, output_prefix::String)
    for (i, sol) in enumerate(niche_solutions)
        output_solution(sol[1], instance, "$(output_prefix)_$i")
    end
    output_solution(merge_solution[1], instance, "$(output_prefix)_merge")
    output_solution(merge_solution[2], instance, "$(output_prefix)_feasible")
end

function plot_all_results(niche_solutions::Vector, merge_solution, output_prefix::String)
    for (i, sol) in enumerate(niche_solutions)
        plot_travels_over_time(sol[7], "$(output_prefix)_$i")
        plot_best_solution_over_time(sol[8], "$(output_prefix)_$i")
    end
    plot_travels_over_time(merge_solution[7], "$(output_prefix)_merge")
    plot_best_solution_over_time(merge_solution[8], "$(output_prefix)_merge")
end

function run_experiment(instance::ProblemInstance, output_prefix::String)
    niche_sols = run_niche_algorithms(instance, output_prefix)
    populations = [sol[3] for sol in niche_sols]
    merge_sol = run_merge_algorithm(instance, output_prefix, populations)
    output_all_solutions(niche_sols, merge_sol, instance, output_prefix)
    plot_all_results(niche_sols, merge_sol, output_prefix)
    return niche_sols, merge_sol
end
