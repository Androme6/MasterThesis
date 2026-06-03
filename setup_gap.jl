include("setup_simulation.jl")

function gap_finder(params::SystemParams, H_fun, F_list, kp, tmax, nframes, filename, save_dir)

    n1_avg = Float64[]
    gap = Float64[]

    if H_fun != H_ideal
        results = get_optimal_frequency(H_fun, params)
        println("Optimal ω2 = ", round(results[1], digits=6))
        println("ω2 dressed = ", round(results[2], digits=6))
        display(results[5])
        params = deepcopy(results[9])
    end

    @showprogress "Sweeping F..." for F in F_list
        
        out = run_simulation(params, H_fun, filename, F, kp, tmax, tmax, nframes, save_dir, false)

        push!(n1_avg, real(out.expect_n1[end]))
       
       
        L_cpu = out.L_cpu 
        
        vals_cpu, _ = eigenstates(
            L_cpu;
            sparse=true,
            sigma = 0.01,
            eigvals=5, 
            krylovdim=30
        )
        
    
        real_parts = sort(real.(vals_cpu), rev=true) 
        
        # real_parts[1] will be the steady state (~ 0.0)
        # real_parts[2] is the true global Liouvillian gap
        current_gap = abs(real_parts[2])
        push!(gap, current_gap)
    
    end

    fig = Figure(size = (800, 600))
    ax = Axis(fig[1, 1], 
        ylabel = L"\text{Liouvillian Gap}", 
        xlabel = L"\text{Steady State }⟨a₁^\dagger a₁⟩", 
        title = L"\text{Liouvillian Gap vs} ⟨a₁^\dagger a₁⟩"
    )
    
    lines!(ax, n1_avg, gap, linewidth = 2, color = :blue)
    CairoMakie.scatter!(ax, n1_avg, gap, markersize = 12, color = :blue)

    CairoMakie.save(save_dir * "\\gap_output.png", fig, px_per_unit = 2) 

    return n1_avg, gap, fig
end
