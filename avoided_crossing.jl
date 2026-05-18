include("setup.jl")
include("setup_resonance_finder.jl")

params = SystemParams(
    ω1 = 1.0, 
    ω2 = 2.0, 
    ωp = 0.0, #2.0, 
    ωq = 2.5, 
    g1 = 0.1, 
    g2 = 0.2,   
    g2p = 0, #0.02,
    θ = π / 6.0
)

results = get_optimal_frequency(H_full, params, ω2_list, ωp_list, lower_index_2, upper_index_2, lower_index_p, upper_index_p)
println("Optimal ω2 = ", round(results[1], digits=6))
println("ω2 dressed = ", round(results[2], digits=6))
println("Optimal ωp = ", round(results[3], digits=6))
println("ωp dressed = ", round(results[4], digits=6))
println("Gap at optimal ω2 = ", round(results[7], digits=6))
println("Gap at optimal ωp = ", round(results[8], digits=6))
display(results[5])  # Plot for ω2 sweep
display(results[6])  # Plot for ωp sweep

results_eff = get_optimal_frequency(H_eff, params, ω2_list, ωp_list, lower_index_2, upper_index_2, lower_index_p, upper_index_p)
println("Optimal ω2 = ", round(results_eff[1], digits=6))
println("ω2 dressed = ", round(results_eff[2], digits=6))
println("Optimal ωp = ", round(results_eff[3], digits=6))
println("ωp dressed = ", round(results_eff[4], digits=6))
println("Gap at optimal ω2 = ", round(results_eff[7], digits=6))
println("Gap at optimal ωp = ", round(results_eff[8], digits=6))
display(results_eff[5])  # Plot for ω2 sweep
display(results_eff[6])  # Plot for ωp sweep