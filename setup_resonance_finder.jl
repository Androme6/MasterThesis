using QuantumToolbox
using CairoMakie
using ProgressMeter
using LaTeXStrings

function get_optimal_frequency(H_fun, p, ω2_lower_bound = 1.9, ω2_upper_bound = 2.2, ωp_lower_bound = 1.7, ωp_upper_bound = 2.0, lower_index_2 = 3, upper_index_2 = 4, lower_index_p = 3, upper_index_p = 5)
    ω2_list = range(ω2_lower_bound * p.ω1, ω2_upper_bound * p.ω1, length=400)
    ωp_list = range(ωp_lower_bound * p.ω1, ωp_upper_bound * p.ω1, length=400)
       
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
    for i in 1:size(eigenvalues, 1)
        lines!(ax2, ω2_list ./ p.ω1, real.(eigenvalues[i, :] .- eigenvalues[1,:]), linewidth=2)
    end
    #finding optimal point
    gap, idx_opt = findmin(real, eigenvalues[upper_index_2, :] - eigenvalues[lower_index_2, :])
    ω2_opt = ω2_list[idx_opt]
    ω2_dressed = real(eigenvalues[upper_index_2, 1] - eigenvalues[1, 1])
    vlines!(ax2, [ω2_opt / p.ω1], color = :black, linestyle = :dash, linewidth = 1.5, label = L"\omega_2^{\text{opt}}")
    hlines!(ax2, [ω2_dressed / p.ω1], color = :red, linestyle = :dash, linewidth = 1.5, label = L"\omega_2^{\text{dressed}}")
    axislegend(ax2, position = :lt)

    #xlims!(ax2, ω2_opt-0.02, ω2_opt+0.02)
    xlims!(ax2, 1.9, 2.2)
    ylims!(ax2, real(eigenvalues[lower_index_2, idx_opt]- eigenvalues[1,idx_opt])-0.05, real(eigenvalues[upper_index_2, idx_opt] - eigenvalues[1,idx_opt])+0.05)

    #=
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
=#
    p_new = deepcopy(p)
    p_new.ω2 = ω2_opt
    #p_new.ωp = ωp_opt
    #p_new.g1p = p_new.g2p * sqrt(p_new.ω1) / sqrt(p_new.ω2)
    

    #return ω2_opt, ω2_dressed, ωp_opt, ωp_dressed, fig2, figp, gap, gap_p, p_new, eigenvalues, ω2_list, eigenvalues_vs_p, ωp_list
    return ω2_opt, ω2_dressed, 0.0, 0.0, fig2, nothing, gap, 0.0, p_new, eigenvalues, ω2_list, 0.0, 0.0
end

function compare(eigenvals_full, eigenvals_eff, eigenvals_num, ω2_list_full, ω2_list_eff, ω2_list_num, ω2_opt_full, ω2_opt_eff, ω2_opt_num, ω2_dressed_full, ω2_dressed_eff, ω2_dressed_num)
    fig_super = Figure(size = (800, 600))
    ax_super = Axis(fig_super[1, 1], 
               xlabel = L"\omega_2/\omega_1", 
               ylabel = L"E/\hbar\omega_1",
               title = "Superimposed Frequency Sweep")
    xlims!(ax_super, (ω2_opt_full + ω2_opt_eff + ω2_opt_num)/3 -0.1, (ω2_opt_full + ω2_opt_eff + ω2_opt_num)/3 +0.1)
    ylims!(ax_super, (ω2_dressed_full + ω2_dressed_eff + ω2_dressed_num)/3 -0.03, (ω2_dressed_full + ω2_dressed_eff + ω2_dressed_num)/3 +0.03)

    for i in 1:size(eigenvals_full, 1)
        # We only add a label to the first line so the legend isn't cluttered with 7 entries
        lbl = i == 1 ? "Full model" : nothing
        lines!(ax_super, ω2_list_full ./ params.ω1, real.(eigenvals_full[i, :] .- eigenvals_full[1, :]), 
           linewidth=2, color=:blue, linestyle=:solid, label=lbl)
    end
    for i in 1:size(eigenvals_eff, 1)
        # We only add a label to the first line so the legend isn't cluttered with 7 entries
        lbl = i == 1 ? "Effective Model" : nothing
        lines!(ax_super, ω2_list_eff ./ params.ω1, real.(eigenvals_eff[i, :] .- eigenvals_eff[1, :]), 
           linewidth=2, color=:red, linestyle=:solid, label=lbl)
    end
    for i in 1:size(eigenvals_num, 1)
        # We only add a label to the first line so the legend isn't cluttered with 7 entries
        lbl = i == 1 ? "Numerical Model" : nothing
        lines!(ax_super, ω2_list_num ./ params.ω1, real.(eigenvals_num[i, :] .- eigenvals_num[1, :]), 
           linewidth=2, color=:green, linestyle=:dash, label=lbl)
    end
    axislegend(ax_super, position=:lt)
    
    return fig_super
end

function analyse_perturbation_validity(H1, H2, params_base, param_type, value_list)
    diff_opt = zeros(length(value_list))
    diff_dressed = zeros(length(value_list))
    gap_1_list = zeros(length(value_list))
    gap_2_list = zeros(length(value_list))
    
    println("Starting validity sweep over ", String(param_type), "...")
    
    @showprogress for i in 1:length(value_list)
        p = deepcopy(params_base)
        val = value_list[i]
        
        if param_type == :ωq
            p.ωq = val
        elseif param_type == :g1
            # Keep the ratio g2/g1 constant while sweeping the overall coupling strength
            ratio = p.g2 / p.g1
            p.g1 = val
            p.g2 = val * ratio
        end
        
        res_1 = get_optimal_frequency(H1, p)
        res_2  = get_optimal_frequency(H2, p)
        
        diff_opt[i] = abs(res_1[1] - res_2[1])
        diff_dressed[i] = abs(res_1[2] - res_2[2])

        gap_1_list[i] = res_1[7]
        gap_2_list[i] = res_2[7]
    end
    
    fig_sweep = Figure(size = (900, 450))
    xlabel_str = param_type == :ωq ? L"\omega_q / \omega_1" : L"g_1 / \omega_1"
    title_prefix = param_type == :ωq ? "Qubit Frequency Sweep: " : "Coupling Strength Sweep: "
    
    ax1 = Axis(fig_sweep[1, 1], 
               xlabel = xlabel_str, 
               ylabel = L"\frac{|\Delta \omega_2^{\text{opt}}|}{\hbar\omega_1}", 
               title = title_prefix * "Error in Optimal ω2 Frequency")
    line_optimal = lines!(ax1, value_list./params.ω1, diff_opt./params.ω1, linewidth=3, color=:purple)
    scatter!(ax1, value_list./params.ω1, diff_opt./params.ω1, color=:purple, markersize=8)
    
    ax1_gap = Axis(fig_sweep[1, 1], yaxisposition = :right,
                   ygridvisible = false, xticklabelsvisible = false, xlabelvisible = false,
                   ylabel = L"\text{Size of Gap } (E/\hbar\omega_1)")
    line_gap_1_1 = lines!(ax1_gap, value_list./params.ω1, gap_1_list./params.ω1, linewidth=2, color=:blue, linestyle=:dash)
    line_gap_1_2 = lines!(ax1_gap, value_list./params.ω1, gap_2_list./params.ω1, linewidth=2, color=:red, linestyle=:dot)
    
    hidespines!(ax1_gap, :l, :t, :b) 
    linkxaxes!(ax1, ax1_gap)

    axislegend(ax1, [line_optimal, line_gap_1_1, line_gap_1_2], ["Error", "Model 1 Gap", "Model 2 Gap"], position=:lt)

    ax2 = Axis(fig_sweep[1, 2], 
               xlabel = xlabel_str, 
               ylabel = L"\frac{|\Delta \omega_2^{\text{dressed}}|}{\hbar\omega_1}", 
               title = title_prefix * "Error in Dressed ω2 Frequency")
    line_dressed = lines!(ax2, value_list./params.ω1, diff_dressed./params.ω1, linewidth=3, color=:darkorange)
    scatter!(ax2, value_list./params.ω1, diff_dressed./params.ω1, color=:darkorange, markersize=8)

    ax2_gap = CairoMakie.Axis(fig_sweep[1, 2], yaxisposition = :right,
                   ygridvisible = false, xticklabelsvisible = false, xlabelvisible = false,
                   ylabel = L"\text{Size of Gap } (E/\hbar\omega_1)")
    line_gap_2_1 = CairoMakie.lines!(ax2_gap, value_list./params.ω1, gap_1_list./params.ω1, linewidth=2, color=:blue, linestyle=:dash)
    line_gap_2_2 = CairoMakie.lines!(ax2_gap, value_list./params.ω1, gap_2_list./params.ω1, linewidth=2, color=:red, linestyle=:dot)
    
    hidespines!(ax2_gap, :l, :t, :b)
    linkxaxes!(ax2, ax2_gap)
    
    # Create a unified legend
    axislegend(ax2, [line_dressed, line_gap_2_1, line_gap_2_2], ["Error", "Model 1 Gap", "Model 2 Gap"], position=:lt)
    colgap!(fig_sweep.layout, 50)
    return fig_sweep, diff_opt, diff_dressed, gap_1_list, gap_2_list
end

function optimise_ωq_g1_landscape(H1, H2, params_base, ωq_list, g1_list)
    N_ωq = length(ωq_list)
    N_g1 = length(g1_list)
    
    error_matrix = zeros(N_ωq, N_g1)
    gap_matrix = zeros(N_ωq, N_g1)
    
    println("Starting 2D Grid Search over ωq (", N_ωq, " points) and g1 (", N_g1, " points)...")
    println("Total evaluations: ", N_ωq * N_g1)
    
    # We use a 1D progress meter over the total iterations
    p = Progress(N_ωq * N_g1, 1, "Sweeping 2D Parameter Space: ")
    
    for i in 1:N_ωq
        for j in 1:N_g1
            p_temp = deepcopy(params_base)
            
            # Set ωq
            p_temp.ωq = ωq_list[i]
            
            # Set g1 and scale g2 accordingly
            ratio = p_temp.g2 / p_temp.g1
            p_temp.g1 = g1_list[j]
            p_temp.g2 = p_temp.g1 * ratio
            
            # Run sweeps
            res_1 = get_optimal_frequency(H1, p_temp, 2.0, 2.1, 1.7, 2.2,3, 4, 3, 5)
            res_2  = get_optimal_frequency(H2, p_temp, 2.0, 2.1, 1.7, 2.2,3, 4, 3, 5)
            
            # Record the discrepancy (using optimal frequency here)
            error_matrix[i, j] = abs(res_1[1] - res_2[1])
            
            # Record the exact gap from the full model
            gap_matrix[i, j] = res_1[7]
            
            next!(p)
        end
    end
    
    # Calculate Figure of Merit: Maximize Gap, Minimize Error
    # Added 1e-6 to prevent division by zero in perfect matches
    FOM_matrix = (gap_matrix.^2) ./ (error_matrix .+ 1e-6)
    
    # Find the indices of the absolute best FOM
    max_idx = argmax(FOM_matrix)
    opt_ωq = ωq_list[max_idx[1]]
    opt_g1 = g1_list[max_idx[2]]
    
    println("\n=== OPTIMIZATION RESULTS ===")
    println("Optimal ωq: ", round(opt_ωq, digits=4))
    println("Optimal g1: ", round(opt_g1, digits=4))
    println("Max FOM: ", round(FOM_matrix[max_idx], digits=2))
    println("Resulting Gap: ", round(gap_matrix[max_idx], digits=5))
    println("Resulting Error: ", round(error_matrix[max_idx], digits=6))
    
    # ==========================================
    # PLOTTING THE 2D HEATMAPS
    # ==========================================
    fig_heat = CairoMakie.Figure(size = (1400, 450))
    
    # Panel 1: The Error
    ax1 = CairoMakie.Axis(fig_heat[1, 1], xlabel = L"\omega_q / \omega_1", ylabel = L"g_1 / \omega_1", title = L"|\Delta \omega_2^{\text{opt}}|")
    hm1 = CairoMakie.heatmap!(ax1, ωq_list./params.ω1, g1_list./params.ω1, error_matrix, colormap = :magma)
    CairoMakie.Colorbar(fig_heat[1, 2], hm1)
    
    # Panel 2: The Gap
    ax2 = CairoMakie.Axis(fig_heat[1, 3], xlabel = L"\omega_q / \omega_1", ylabel = L"g_1 / \omega_1", title = "Full Model Gap Size")
    hm2 = CairoMakie.heatmap!(ax2, ωq_list, g1_list, gap_matrix, colormap = :viridis)
    CairoMakie.Colorbar(fig_heat[1, 4], hm2)
    
    # Panel 3: Figure of Merit
    ax3 = CairoMakie.Axis(fig_heat[1, 5], xlabel = L"\omega_q / \omega_1", ylabel = L"g_1 / \omega_1", title = "Figure of Merit (Gap / Error)")
    hm3 = CairoMakie.heatmap!(ax3, ωq_list, g1_list, FOM_matrix, colormap = :plasma)
    CairoMakie.Colorbar(fig_heat[1, 6], hm3)
    
    # Add a star marker at the optimal point on the FOM plot!
    CairoMakie.scatter!(ax3, [opt_ωq], [opt_g1], marker = :star5, markersize = 20, color = :cyan, strokecolor = :black, strokewidth = 1)
    
    return fig_heat, opt_ωq, opt_g1, FOM_matrix, gap_matrix
end