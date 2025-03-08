include("utilitaries.jl")
include("structures.jl")


function plot_patient_positions(instance::ProblemInstance)
    x = [instance.patients[string(patient)]["x_coord"] for patient in 1:length(instance.patients)]
    y = [instance.patients[string(patient)]["y_coord"] for patient in 1:length(instance.patients)]
    labels = [string(patient) for patient in 1:length(instance.patients)]
    scatter(x, y, labels=labels, title="Patient positions", xlabel="x", ylabel="y")
end

function plot_routes(individual::Individual, instance::ProblemInstance)
    amount_of_unempty_routes = length([route for route in individual.routes if !isempty(route)])
    plot()
    x = [instance.patients[string(patient)]["x_coord"] for patient in 1:length(instance.patients)]
    y = [instance.patients[string(patient)]["y_coord"] for patient in 1:length(instance.patients)]
    scatter!(x, y, title="Patient positions")
    depot_coords = (instance.depot["x_coord"], instance.depot["y_coord"])
    scatter!([depot_coords[1]], [depot_coords[2]], ms=5)
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
        plot!(x, y, ms=5, legend=false)
    end
    display(plot!(title="Routes"))
    println("Plotting $(amount_of_unempty_routes)) routes for a travel time of $(individual.total_travel_time)")
end

function plot_travels_over_time(travel_times)
    display(plot(travel_times))
end

function output_solution(individual::Individual, instance::ProblemInstance, filename::String)
    open(filename, "w") do io
        # Écrire la capacité de l'infirmière et le temps de retour au dépôt
        println(io, "Nurse capacity : $(instance.capacity_nurse)")
        println(io, "Depot return time : $(instance.depot["return_time"])")
        println(io, "-"^100)

        # Parcourir chaque route de l'individu
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

            # Ajouter le retour au dépôt
            travel_time = get_travel_time(last_patient_id, 0, instance)
            arrival_time = elapsed_time + travel_time
            route_str *= "D($(arrival_time))"
            route_duration += travel_time

            # Écrire les informations de la route
            println(io, "Nurse $(i) (N$(i))\t$(route_duration)\t$(nurse_demand)\t$(route_str)")
        end

        println(io, "-"^100)
        println(io, "Objective value (total duration): $(individual.total_travel_time)")
    end
end