include("setup_simulation.jl")

function gap_finder(params::SystemParams, H_fun, F_list, tmax, nframes, filename, save_dir)
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
        
        p = deepcopy(params)
        p.F = F

        out = run_simulation(p, H_fun, filename, tmax, tmax, nframes, save_dir, false)
        
        push!(n1_avg, out.expect_n1[end])
        println("  -> Steady State ⟨a₁† a₁⟩ = ", out.expect_n1[end])
        
        vals, _ = eigenstates(
                    out.L_cpu; 
                    sparse = true, 
                    sigma = 1e-5im,      # Purely imaginary shift hides ghosts from the real axis
                    eigvals = 20, 
                    krylovdim = 100,     # Increased to give the solver more breathing room
                    tol = 1e-12          # Stricter tolerance forces higher accuracy
                )

        real_parts = sort(real.(vals), rev=true)

        current_gap = NaN
        for val in real_parts
            if val < -1e-15
                current_gap = abs(val)
                break
            end
        end
        
        push!(gap, current_gap)
        println("  -> True Bit-Flip Gap: ", current_gap)
    end

    # Plotting
    fig = Figure(size = (800, 600))
    ax = Axis(fig[1, 1], 
        ylabel = L"\text{Liouvillian Gap}", 
        xlabel = L"\text{Steady State }⟨a₁^\dagger a₁⟩", 
        title = L"\text{Exponential Closing Gap}",
        yscale = log10
    )
    lines!(ax, n1_avg, gap, linewidth = 2.5, color = :darkred)
    CairoMakie.scatter!(ax, n1_avg, gap, markersize = 12, color = :darkred)
    CairoMakie.save(save_dir * "\\gap_output.png", fig, px_per_unit = 2) 
    display(fig)
    
    return  n1_avg, gap, fig
end

