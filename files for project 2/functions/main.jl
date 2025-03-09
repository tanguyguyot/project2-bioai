include("genetic_algo2.jl")
include("plotting.jl")

Random.seed!(1234)

# 70 generations a 500 individus : 92 s
# sol is (best_solution0, population0, fitnesses0, scores0, average_fitnesses0, travel_times0)
# sol9_twc = @time genetic_algorithm(data_instance9, [0, 0, 1.0], 20, 100, 3, 4, 0.15, 50.0)

# NICHING ON SOL9
@time begin
    @time sol9_1 = genetic_algorithm(data_instance9, [0.3, 0.7, 0], 50, 125, 3, 4, 0.15, 300.0, "output9_1")
    @time sol9_2 = genetic_algorithm(data_instance9, [0.7, 0.3, 0], 50, 125, 3, 4, 0.15, 300.0, "output9_2")
    @time sol9_3 = genetic_algorithm(data_instance9, [0.0, 0.0, 1.0], 50, 125, 3, 4, 0.15, 300.0, "output9_3")
    @time sol9 = genetic_algorithm(data_instance9, [0.5, 0.5, 0], 30, 375, 3, 4, 0.15, 500.0, "output9", true, [sol9_1[3], sol9_2[3], sol9_3[3]])
plot(sol9[6])
end
#=

@time begin
    # sol0_1 = genetic_algorithm(data_instance0, [0.3, 0.7, 0], 50, 100, 3, 4, 0.15, 300.0, "output0_1")
    # sol0_2 = genetic_algorithm(data_instance0, [0.7, 0.3, 0], 50, 100, 3, 4, 0.15, 300.0, "output0_2")
    # sol0_3 = genetic_algorithm(data_instance0, [0.0, 0.0, 1.0], 50, 100, 3, 4, 0.15, 300.0, "output0_3")
    # time for sol0 (25, 300) : approx. 8 min ; score : 828.9368669428338 (best, near 826 benchmark)
    @time sol0 = genetic_algorithm(data_instance0, [0.5, 0.5, 0], 25, 300, 3, 4, 0.15, 500.0, "output0", true, [sol0_1[3], sol0_2[3], sol0_3[3]])
end

plot(sol0[5], label="Average fitness")
plot!(sol0[7], label="Best solution fitness")



# total time : 568 s
@time begin
    # time :
    #@time sol1_1 = genetic_algorithm(data_instance1, [0.3, 0.7, 0], 50, 100, 3, 4, 0.15, 300.0, "output1_1")
    # time : 
    #@time sol1_2 = genetic_algorithm(data_instance1, [0.7, 0.3, 0], 50, 100, 3, 4, 0.15, 300.0, "output1_2")
    # time :
    #@time sol1_3 = genetic_algorithm(data_instance1, [0.0, 0.0, 1.0], 50, 100, 3, 4, 0.15, 300.0, "output1_3")
    # time : 212 s ; score of 1659.1448976431811 (je pense on peut mettre plus d'itérations) ; benchmark 1514 donc on est entre 5 et 10%
    @time sol1 = genetic_algorithm(data_instance1, [0.5, 0.5, 0], 30, 300, 3, 4, 0.15, 500.0, "output1", true, [sol1_1[3], sol1_2[3], sol1_3[3]])
end

plot(sol1[5], label="Average fitness")
plot!(sol1[7], label="Best solution fitness")
output_all_solution(sol1, "output11")
=#

# time : 4422 s soit 73 min (une heure 13 min)
#=
@time begin
    println("Data instance 2, niche 1")
    @time sol2_1 = genetic_algorithm(data_instance2, [0.3, 0.7, 0], 50, 150, 3, 4, 0.15, 300.0, "output2_1")
    println("Data instance 2, niche 2")
    #time : 739 s ; score 1013
    @time sol2_2 = genetic_algorithm(data_instance2, [0.7, 0.3, 0], 50, 150, 3, 4, 0.15, 300.0, "output2_2")
    println("Data instance 2, niche 3")
    # time : 739 s ; score : 959.971987
    @time sol2_3 = genetic_algorithm(data_instance2, [0.0, 0.0, 1.0], 50, 150, 3, 4, 0.15, 300.0, "output2_3")
    println("Data instance 2, island merge")
    # time : 1153 s ; 1002
    @time sol2 = genetic_algorithm(data_instance2, [0.5, 0.5, 0], 30, 450, 3, 4, 0.15, 500.0, "output2", true, [sol2_1[3], sol2_2[3], sol2_3[3]])
    # time : 1789s ; score 955.7316093300215
end

plot(sol2[5], label="Average fitness")
plot!(sol2[7], label="Best solution fitness")
output_all_solution(sol2, "output20")

=#
