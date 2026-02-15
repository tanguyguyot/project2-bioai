include("utilitaries.jl")
include("structures.jl")

# Cluster patients based on their position ; cluster number is random between 12 and 25
function position_cluster_solution(instance::ProblemInstance)::Individual
    nb_patients = length(instance.patients)
    nb_nurses = instance.nbr_nurses
    # random amount of clusters
    nb_clusters = rand(div(nb_nurses, 4):div(nb_nurses, 2))
    coordinates = hcat([instance.patients[string(patient)]["x_coord"] for patient in 1:nb_patients], [instance.patients[string(patient)]["y_coord"] for patient in 1:nb_patients])'
    R = kmeans(coordinates, nb_clusters)
    clusters = R.assignments
    routes = [[] for _ in 1:nb_nurses]
    for cluster in 1:nb_clusters
        patients = findall(x -> x == cluster, clusters)
        routes[cluster] = patients
    end
    return Individual(routes, 0.0, 0.0, 0.0, false)
end

# Cluster patients based on their time window
function time_window_cluster_solution(instance::ProblemInstance)::Individual
    patients = instance.patients
    nb_patients = length(patients)
    nb_nurses = instance.nbr_nurses
    # random amount of clusters
    nb_clusters = rand(div(nb_nurses, 4):div(nb_nurses, 2))
    data = [ [(patients[string(patient)]["start_time"] + patients[string(patient)]["end_time"]) / 2, patients[string(patient)]["end_time"] - patients[string(patient)]["start_time"]] for patient in 1:nb_patients]
    data_matrix = hcat(data...)
    R = kmeans(data_matrix, nb_clusters)
    clusters = R.assignments
    clusters_and_idx = [(i, clusters[i]) for i in eachindex(clusters)]
    routes = [[] for _ in 1:nb_nurses]

    for route in routes
        for cluster in 1:nb_clusters
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
    # if remaining clusters, add them at the end of the route
    if !isempty(clusters_and_idx)
        push!(routes, [i[1] for i in clusters_and_idx])
    end
    return Individual(routes, 0.0, 0.0, 0.0, false)
end

# Completely random solution
function random_solution(instance::ProblemInstance)::Individual
    nb_patients = length(instance.patients)
    nb_nurses = instance.nbr_nurses
    nb_cluster = rand(div(nb_nurses, 4):div(nb_nurses, 2))
    permutation = randperm(nb_patients)
    separations = sort(sample(1:nb_patients, nb_cluster-1, replace=false))
    finale_routes = [[] for _ in 1:nb_nurses]
    route_counter = 1 # de 1 à 25
    for i in 1:nb_patients
        if route_counter == nb_cluster #last route
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
        return vcat(populations...)
    end
    return vcat([random_solution(instance) for _ in 1:(population_size * proportion[1])], 
                [position_cluster_solution(instance) for _ in 1:population_size * proportion[2]],
                [time_window_cluster_solution(instance) for _ in 1:population_size * proportion[3]])
end
