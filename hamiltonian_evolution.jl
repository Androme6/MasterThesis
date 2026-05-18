include("setup.jl")
include("setup_resonance_finder.jl")


function hamiltonian_evolution(H, ωd, F, t, S)
    H_drive_bare = -F * 1im * (a2' - a2) #for now, drive on mode 2
    H_drive_eff = H_drive_bare + commutator(S, H_drive_bare)
    drive_func(p, t) = cos(ωd * t)
    H_tot = H + QobjEvo(H_drive_eff, drive_func)
    psi0_dressed = fock(N1*N2*Np*Nq, 0; dims = dims_sys)
    sol = sesolve(H_tot, psi0_dressed, t, maxiters=1e12)
   
    states_cpu_mats = [ket2dm(state).data for state in sol.states]
    V_mat = exp(Matrix(-S.data))
    rho_mode1, W, xvec, yvec, idx = calculate_wigner(states_cpu_mats, V_mat, t, t[end], ωd)

    fig_wigner = Figure(size = (700, 600))
    ax_wigner = Axis(fig_wigner[1, 1], 
        title = "Wigner Function (Memory Mode)", 
        xlabel = "Re(α)", 
        ylabel = "Im(α)", 
        aspect = 1
    )
    hm = heatmap!(ax_wigner, xvec, yvec, W, colormap = :RdBu)
    Colorbar(fig_wigner[1, 2], hm, label = "W(α)")
    return sol, fig_wigner
end

params = SystemParams(
    ω1 = 1.0, 
    ω2 = 2.0, 
    ωp = 100.0, 
    ωq = 2.5, 
    g1 = 0.1, 
    g2 = 0.2,   
    g2p = 0, 
    θ = π / 6.0
)

F = 0.065
tmax = 35000
steps = 2000
t = LinRange(0, tmax, steps)

results_full = get_optimal_frequency(H_full, params, ω2_list, ωp_list, lower_index_2, upper_index_2, lower_index_p, upper_index_p)
results_eff = get_optimal_frequency(H_eff, params, ω2_list, ωp_list, lower_index_2, upper_index_2, lower_index_p, upper_index_p)
results_num = get_optimal_frequency(H_num, params, ω2_list, ωp_list, lower_index_2, upper_index_2, lower_index_p, upper_index_p)
Hfull = H_full(results_full[9])
Heff = H_eff(results_eff[9])
Seff = SW_generator(results_eff[9])
Hnum = H_num(results_num[9])
Snum = SW_generator(results_num[9])
println("Optimal ω2 (full)= ", round(results_full[1], digits=6))
println("ω2 dressed (full)= ", round(results_full[2], digits=6))
println("Optimal ω2 (eff)= ", round(results_eff[1], digits=6))
println("ω2 dressed (eff)= ", round(results_eff[2], digits=6))
println("Optimal ω2 (num)= ", round(results_num[1], digits=6))
println("ω2 dressed (num)= ", round(results_num[2], digits=6))


sol_full, fig_wigner_full = hamiltonian_evolution(Hfull, results_full[2], F, t, 0.0*Id)
sol_eff, fig_wigner_eff = hamiltonian_evolution(Heff, results_eff[2], F, t, Seff)
sol_num, fig_wigner_num = hamiltonian_evolution(Hnum, results_num[2], F, t, Snum)
display(fig_wigner_full)
display(fig_wigner_eff)
display(fig_wigner_num)