struct ProblemInstance
    depot::Dict{String, Any}
    patients::Dict{String, Any}
    nbr_nurses::Int
    instance_name::String
    travel_times::Array{Array{Float64,1},1}
    benchmark::Float64
    capacity_nurse::Int
    travel_time_matrix::Vector{Any}
end

mutable struct Individual
    routes::Vector{Vector{Int}}  # Each route is a list of patient indices
    total_travel_time::Float64    # The total travel time
    score::Float64                # The total score of the solution
    penalties::Float64           # The total penalties of the solution
    feasability::Bool
end