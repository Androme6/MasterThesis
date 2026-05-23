include("setup.jl")
include("setup_resonance_finder.jl")

params = SystemParams(
    ω1 = 1.0, 
    ω2 = 2.0, 
    ωp = 0.0, #2.0, 
    ωq = 2.5,#2.5, 
    g1 = 0.1,#0.1, 
    g2 = 0.2,#0.2,   
    g2p = 0, #0.02,
    θ = π / 6.0,
    ωd = 0.0
)

results_full = get_optimal_frequency(H_full, params)
println("Optimal ω2 (full) = ", round(results_full[1], digits=12))
println("ω2 dressed (full) = ", round(results_full[2], digits=12))
#println("Optimal ωp (full) = ", round(results_full[3], digits=6))
#println("ωp dressed (full) = ", round(results_full[4], digits=6))
println("Gap at optimal ω2 (full) = ", round(results_full[7], digits=12))
#println("Gap at optimal ωp (full) = ", round(results_full[8], digits=6))
display(results_full[5])  # Plot for ω2 sweep
#display(results_full[6])  # Plot for ωp sweep

results_eff = get_optimal_frequency(H_eff_4th_order_RWA, params)
println("Optimal ω2 (eff) = ", round(results_eff[1], digits=12))
println("ω2 dressed (eff) = ", round(results_eff[2], digits=12))
#println("Optimal ωp (eff) = ", round(results_eff[3], digits=6))
#println("ωp dressed (eff) = ", round(results_eff[4], digits=6))
println("Gap at optimal ω2 (eff) = ", round(results_eff[7], digits=12))
#println("Gap at optimal ωp (eff) = ", round(results_eff[8], digits=6))
display(results_eff[5])  # Plot for ω2 sweep
#display(results_eff[6])  # Plot for ωp sweep

results_num = get_optimal_frequency(H_num, params)
println("Optimal ω2 (num) = ", round(results_num[1], digits=12))
println("ω2 dressed (num) = ", round(results_num[2], digits=12))
#println("Optimal ωp (num) = ", round(results_num[3], digits=6))
#println("ωp dressed (num) = ", round(results_num[4], digits=6))
println("Gap at optimal ω2 (num) = ", round(results_num[7], digits=12))
#println("Gap at optimal ωp (num) = ", round(results_num[8], digits=6))
display(results_num[5])  # Plot for ω2 sweep
#display(results_num[6])  # Plot for ωp sweep

fig_compare = compare(results_full[10], results_eff[10], results_num[10], results_full[11], results_eff[11], results_num[11], results_full[1], results_eff[1], results_num[1], results_full[2], results_eff[2], results_num[2])
display(fig_compare)










##

ωq_sweep_list = range(2.5, 3.5, length=12)
fig_ωq, d_optimal_ωq, d_dressed_ωq, gap_full_list_ωq, gap_eff_list_ωq = analyse_perturbation_validity(H_full, H_eff, params, :ωq, ωq_sweep_list)
display(fig_ωq)

g1_sweep_list = range(0.01, 0.1, length=12)
fig_g1, d_optimal_g1, d_dressed_g1, gap_full_list_g1, gap_eff_list_g1 = analyse_perturbation_validity(H_full, H_eff, params, :g1, g1_sweep_list)
display(fig_g1)

##

ωq_grid = range(3.3, 3.5, length=10)  # Try length=10 or 15 later
g1_grid = range(0.05, 0.8, length=10) # Try length=10 or 15 later

fig_2d, best_ωq, best_g1, fom_data, gap_data = optimise_ωq_g1_landscape(H_full, H_eff, params, ωq_grid, g1_grid)

display(fig_2d)