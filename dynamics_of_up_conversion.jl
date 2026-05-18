include("setup.jl")
include("setup_resonance_finder.jl")

function Rabi_Oscillations(H, t, p)
    psi_bare_200g = tensor(fock(N1, 2), fock(N2, 0), basis(Np, 0),basis(Nq, 1))
    psi_bare_010g = tensor(fock(N1, 0), fock(N2, 1), basis(Np, 0),basis(Nq, 1))
    psi_bare_001g = tensor(fock(N1, 0), fock(N2, 0), basis(Np, 1),basis(Nq, 1))

    p_off = deepcopy(p)
    p_3WM = deepcopy(p)
    p_filter = deepcopy(p)

    
    p_off.ω2 -= 0.3 
    #p_off.ωp += 0.2
    H_off = H(p_off)
    _, ψ, _ = eigenstates(H_off)
    idx_200g = findmax(vi -> fidelity(vi, psi_bare_200g), ψ[1:10])[2]
    idx_010g = findmax(vi -> fidelity(vi, psi_bare_010g), ψ[1:10])[2]
    idx_001g = findmax(vi -> fidelity(vi, psi_bare_001g), ψ[1:10])[2]
    P_dressed_200g = ket2dm(ψ[idx_200g])
    P_dressed_010g = ket2dm(ψ[idx_010g])
    P_dressed_001g = ket2dm(ψ[idx_001g])

    # 3WM
    #p_3WM.g1p = 0
    #p_3WM.g2p = 0
    H_3WM = H(p_3WM)    
    sol_3WM = sesolve(H_3WM, ψ[idx_200g], t, e_ops=[P_dressed_200g, P_dressed_010g])

    #filter
    p_filter.ω1 -= 0.3
    H_filter = H(p_filter)
    sol_filter = sesolve(H_filter, ψ[idx_001g], t, e_ops=[P_dressed_010g, P_dressed_001g])


   # Create the first figure (3WM)
    fig_3wm = Figure()
    ax_3wm = Axis(fig_3wm[1, 1], 
        xlabel = L"Time $(1/\omega_1)$", 
        ylabel = "Population",
        title = L"On Resonance $(\omega_2 \approx %$(round(p_3WM.ω2, digits=3)))$"   
    )
    # Plot the lines onto the 3WM axis
    lines!(ax_3wm, t, real.(sol_3WM.expect[1, :]), label=L"|2, 0, 0, g\rangle", linewidth=2, color=:blue)
    lines!(ax_3wm, t, real.(sol_3WM.expect[2, :]), label=L"|0, 1, 0, g\rangle", linewidth=2, color=:orange)  # Display the legend
    axislegend(ax_3wm)


    # Create the second figure (Filter)
    fig_filter = Figure()
    ax_filter = Axis(fig_filter[1, 1], 
        xlabel = L"Time $(1/\omega_1)$", 
        ylabel = "Population",
        title = L"On Resonance $(\omega_p \approx %$(round(p_filter.ωp, digits=3)))$"
    )
    # Plot the lines onto the Filter axis
    lines!(ax_filter, t, real.(sol_filter.expect[1, :]), label=L"|0, 1, 0, g\rangle", linewidth=2, color=:blue)
    lines!(ax_filter, t, real.(sol_filter.expect[2, :]), label=L"|0, 0, 1, g\rangle", linewidth=2, color=:orange)
    axislegend(ax_filter)

    return sol_3WM, sol_filter, fig_3wm, fig_filter
end

params = SystemParams(
    ω1 = 1.0, 
    ω2 = 2.0, 
    ωp = 0.0, 
    ωq = 2.5, 
    g1 = 0.1, 
    g2 = 0.2,   
    g2p = 0, 
    θ = π / 6.0
)
tmax = 1000
steps = 200
t = LinRange(0, tmax, steps)

results_full = get_optimal_frequency(H_full, params, ω2_list, ωp_list, lower_index_2, upper_index_2, lower_index_p, upper_index_p)
results_eff = get_optimal_frequency(H_eff, params, ω2_list, ωp_list, lower_index_2, upper_index_2, lower_index_p, upper_index_p)


sol_3WM, sol_filter, fig_3wm, fig_filter = Rabi_Oscillations(H_full, t, results_full[9])
sol_3WM_eff, sol_filter_eff, fig_3wm_eff, fig_filter_eff = Rabi_Oscillations(H_eff, t, results_eff[9])
display(fig_3wm)
display(fig_filter)
display(fig_3wm_eff)
display(fig_filter_eff)