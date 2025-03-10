include("utilitaries.jl")
include("structures.jl")


function plot_patient_positions(instance::ProblemInstance)
    x = [instance.patients[string(patient)]["x_coord"] for patient in 1:length(instance.patients)]
    y = [instance.patients[string(patient)]["y_coord"] for patient in 1:length(instance.patients)]
    labels = [string(patient) for patient in 1:length(instance.patients)]
    scatter(x, y, labels=labels, title="Patient positions", xlabel="x", ylabel="y")
end

function plot_routes(individual::Individual, instance::ProblemInstance, path_name::String)
    amount_of_unempty_routes = length([route for route in individual.routes if !isempty(route)])
    
    colors = [
        :red, :blue, :green, :orange, :purple, 
        :pink, :yellow, :cyan, :magenta, :lime, 
        :indigo, :brown, :navy, :maroon, :gold, 
        :violet, :black, :coral, :teal, :khaki, 
        :salmon, :turquoise, :lavender, :white, :gray, :red, :blue, :green, :orange
    ]
    
    plot()
    x = [instance.patients[string(patient)]["x_coord"] for patient in 1:length(instance.patients)]
    y = [instance.patients[string(patient)]["y_coord"] for patient in 1:length(instance.patients)]
    scatter!(x, y, title="Patient positions")
    for (i, route) in enumerate(individual.routes)
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
        plot!(x, y, ms=5, legend=false, color=colors[i])
    end
    depot_coords = (instance.depot["x_coord"], instance.depot["y_coord"])
    scatter!([depot_coords[1]], [depot_coords[2]], ms=5)
    plot!(title="Routes")
    println("Plotting $(amount_of_unempty_routes)) routes for a travel time of $(individual.total_travel_time) \n")
    savefig("./plots/$path_name")
end

function plot_travels_over_time(travel_times, output_name::String)
    plot()
    display(plot!((travel_times), title="Travel time over generations", xlabel="Generation", ylabel="Travel time", legend=false, color=:blue))
    savefig("./plots/$(output_name)_travels_over_time.png")
end

function plot_best_solution_over_time(list, output_name::String)
    plot()
    scores = [x.score for x in list]
    display(plot!(scores, color=:orange, sm=3, title="Best score over generations", xlabel="Generation", ylabel="Score", legend=false))
    savefig("./plots/$(output_name)_best_sol_over_time.png")
end