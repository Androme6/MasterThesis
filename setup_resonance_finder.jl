using QuantumToolbox
using CairoMakie
using ProgressMeter
using LaTeXStrings

function get_optimal_frequency(H_fun, p, ω2_list, ωp_list, lower_index_2, upper_index_2, lower_index_p, upper_index_p)
    #ω2 part
    #ωp = 2.0*p.ω1 + 10.0*p.g2p
    #p.ωp = ωp
    #sweep
    eigenvalues = @showprogress mapreduce(hcat, ω2_list) do ω2
            p.ω2 = ω2
            p.g1p = p.g2p * sqrt(p.ω1) / sqrt(p.ω2)
            H = H_fun(p)
            eigenstates(H, sparse = true, sigma = -p.ωq, eigvals = 7).values
    end
    #plotting
    fig2 = Figure(size = (800, 600))
    ax2 = Axis(fig2[1, 1], 
               xlabel = L"\omega_2/\omega_1", 
               ylabel = L"E/\hbar\omega_1")
    ylims!(ax2, 1.9, 2.05)
    for i in 1:size(eigenvalues, 1)
        lines!(ax2, ω2_list ./ p.ω1, real.(eigenvalues[i, :] .- eigenvalues[1,:]), linewidth=2)
    end
    #finding optimal point
    gap, idx_opt = findmin(real, eigenvalues[upper_index_2, :] - eigenvalues[lower_index_2, :])
    ω2_opt = ω2_list[idx_opt]
    ω2_dressed = real(eigenvalues[upper_index_2, 1] - eigenvalues[1, 1])

    #ωp part
    p.ω2 = ω2_opt
    p.g1p = p.g2p * sqrt(p.ω1) / sqrt(p.ω2)
    #sweep
    eigenvalues_vs_p = @showprogress mapreduce(hcat, ωp_list) do ωp
            p.ωp = ωp
            H = H_fun(p)
            eigenstates(H, sparse = true, sigma = -p.ωq, eigvals = 7).values
    end
    #plotting
    figp = Figure(size = (800, 600))
    axp = Axis(figp[1, 1], 
               xlabel = L"\omega_p/\omega_1", 
               ylabel = L"E/\hbar\omega_1")
    ylims!(axp, 1.9, 2.05)
    for i in 1:size(eigenvalues_vs_p, 1)
        lines!(axp, ωp_list ./ p.ω1, real.(eigenvalues_vs_p[i, :] .- eigenvalues_vs_p[1,:]), linewidth=2)
    end
    #finding optimal point
    gap_p, idx_opt_p = findmin(real, eigenvalues_vs_p[upper_index_p, :] - eigenvalues_vs_p[lower_index_p, :])
    ωp_opt = ωp_list[idx_opt_p]
    ωp_dressed = real(eigenvalues_vs_p[upper_index_p, 1] - eigenvalues_vs_p[1, 1])

    p_new = deepcopy(p)
    p_new.ω2 = ω2_opt
    #p_new.ωp = ωp_opt
    #p_new.g1p = p_new.g2p * sqrt(p_new.ω1) / sqrt(p_new.ω2)

    return ω2_opt, ω2_dressed, ωp_opt, ωp_dressed, fig2, figp, gap, gap_p, p_new
end

ω2_list = range(1.9 * params.ω1, 2.1 * params.ω1, length=300)
ωp_list = range(1.7 * params.ω1, 2.2 * params.ω1, length=300)
lower_index_2 = 3
upper_index_2 = 4
lower_index_p = 3
upper_index_p = 5