using QuantumToolbox
using CairoMakie
using LaTeXStrings
using Dates
using LinearAlgebra

function calculate_occupations(states_cpu_mats, V_mat)
    println("Calculating occupations...")
    len = length(states_cpu_mats)
    expect_n1_plot = zeros(len)
    expect_n2_plot = zeros(len)
    expect_np_plot = zeros(len)

    n1_dressed = V_mat' * Array(n1.data) * V_mat
    n2_dressed = V_mat' * Array(n2.data) * V_mat
    np_dressed = V_mat' * Array(np.data) * V_mat

    for i in 1:len
        rho = states_cpu_mats[i]
        expect_n1_plot[i] = real(dot(n1_dressed', rho))
        expect_n2_plot[i] = real(dot(n2_dressed', rho))
        expect_np_plot[i] = real(dot(np_dressed', rho))
    end

    return expect_n1_plot, expect_n2_plot, expect_np_plot
end


function extract_memory_state(states_cpu_mats, V_mat, t, t_selected, ωd)
    println("Extracting memory counter-rotated density matrix...")
    t_selected_idx = argmin(abs.(t .- t_selected))
    rho_final_dressed_mat = states_cpu_mats[t_selected_idx]
    rho_final_bare_mat = V_mat * rho_final_dressed_mat * V_mat'
    t_final = t[t_selected_idx]
    U_rot = exp(1im * ωd/2 * t_final * Array(n1.data))
    rho_final_bare_rotated_mat = U_rot * rho_final_bare_mat * U_rot'
    rho_final_bare_rotated_qobj = QuantumObject(rho_final_bare_rotated_mat, type=Operator(), dims=dims_sys)
    rho_mode1_rotated = ptrace(rho_final_bare_rotated_qobj, 1)

    return rho_mode1_rotated, t_selected_idx
end

function calculate_wigner(states_cpu_mats, V_mat, t, t_selected, ωd)
    println("Calculating Wigner function...")

    rho_mode1_rotated, t_selected_idx = extract_memory_state(states_cpu_mats, V_mat, t, t_selected, ωd)
    
    xvec = LinRange(-5, 5, 100)
    yvec = LinRange(-5, 5, 100)
    W_cat = wigner(rho_mode1_rotated, xvec, yvec)'
    
    return rho_mode1_rotated, W_cat, xvec, yvec, t_selected_idx
end

function plotting(t, expect_n1, expect_n2, expect_np, rho_mode1_rotated, W_cat, xvec, yvec, t_selected_idx, save_dir, filename)
    println("Generating plots...")
    fig_master = Figure(size = (1200, 1000))

    # Panel 1: Population Dynamics
    ax_pop = Axis(fig_master[1, 1], title="Occupation Numbers", xlabel=L"\text{Time}  (1/\omega_1)", ylabel="Average Occupation Number")
    lines!(ax_pop, t, expect_np, label="<nP> (Purcell Filter)", linewidth=3, color=:green)
    lines!(ax_pop, t, expect_n2, label="<n2> (Buffer Mode)", linewidth=3, color=:orange)
    lines!(ax_pop, t, expect_n1, label="<n1> (Memory Mode)", linewidth=3, color=:blue)
    axislegend(ax_pop, position=:lt)

    # Panel 2: Fock State Populations Histogram
    fock_populations = real.(diag(rho_mode1_rotated.data))
    N_dim = size(rho_mode1_rotated.data, 1) # Dynamically fetch dimension instead of relying on global N1
    photon_numbers = 0:(N_dim-1)
    
    ax_fock = Axis(fig_master[1, 2], title="Fock Populations, t = $(round(t[t_selected_idx], digits=2))", xlabel="Photon Number (n)", ylabel="Probability P(n)")
    barplot!(ax_fock, photon_numbers, fock_populations, color=:dodgerblue, strokecolor=:black, strokewidth=1)
    CairoMakie.xlims!(ax_fock, -0.5, 30)

    # Panel 3: 2D Wigner Function
    ax2D = Axis(fig_master[2, 1], title = "2D Wigner, t = $(round(t[t_selected_idx], digits=2))", xlabel = "Re(α)", ylabel = "Im(α)", aspect = 1) 
    hm = CairoMakie.heatmap!(ax2D, xvec, yvec, W_cat, colormap = :RdBu)
    Colorbar(fig_master[2, 1, Right()], hm, label = "W(α)")

    # Panel 4: 3D Wigner Function
    ax3D = Axis3(fig_master[2, 2], title = "3D Wigner, t = $(round(t[t_selected_idx], digits=2))", xlabel = "Re(α)", ylabel = "Im(α)", zlabel = "W(α)", elevation = pi/6, azimuth = pi/4) 
    CairoMakie.surface!(ax3D, xvec, yvec, W_cat, colormap = :RdBu)
    CairoMakie.colgap!(fig_master.layout, 150)

    # Save Plot
    save_path_img = joinpath(save_dir, filename * ".png")
    CairoMakie.save(save_path_img, fig_master, px_per_unit = 2) 
    println("Saved image to: ", save_path_img)

    return fig_master
end

function text_summary(params, expect_n1, expect_n2, expect_np, ωd, F, kp, save_dir, filename)
    println("Saving text logs...")
    summary_text = """
    --- System Parameters ---
    ω1 = $(params.ω1) | ω2 = $(params.ω2) | ωp = $(params.ωp) | ωq  = $(params.ωq)
    g1 = $(params.g1) | g2 = $(params.g2) | g1p = $(params.g1p) | g2p = $(params.g2p)
    θ    = $(round(params.θ, digits=3))
    κp  = $(kp) | F = $(F) | ωd  = $(round(ωd, digits=5))

    Observables:
    Final ⟨n1⟩ = $(round(expect_n1[end], digits=4))
    Final ⟨n2⟩ = $(round(expect_n2[end], digits=4))
    Final ⟨nP⟩ = $(round(expect_np[end], digits=4))
    """
    save_path_txt = joinpath(save_dir, filename * ".txt")
    open(save_path_txt, "w") do file
        write(file, summary_text)
    end
    println("Saved logs to:  ", save_path_txt)
end


# Main function to run analysis and generate plots
function analysis_and_plots(states_cpu_mats, V_mat, t, t_selected, params, expect_n1, expect_n2, expect_np, ωd, F, kp, save_dir, filename)
    # Calculate Wigner
    rho_mode1_rotated, W_cat, xvec, yvec, t_selected_idx = calculate_wigner(states_cpu_mats, V_mat, t, t_selected, ωd)
    
    # Generate and Save Plots
    fig_master = plotting(t, expect_n1, expect_n2, expect_np, rho_mode1_rotated, W_cat, xvec, yvec, t_selected_idx, save_dir, filename)
    
    # Generate and Save Text Logs
    text_summary(params, expect_n1, expect_n2, expect_np, ωd, F, kp, save_dir, filename)
    return fig_master
end