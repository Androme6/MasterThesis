include("setup.jl")
include("setup_resonance_finder.jl")

function MonitorRegression(current_errors; filename="previous_errors.txt")
    println("\n==========================================")
    println(" Regression Monitor")
    println("==========================================")

    if isfile(filename)
        # Read plain text file line by line and parse back into numbers
        raw_lines = readlines(filename)
        previous_errors = [parse(Float64, line) for line in raw_lines if strip(line) != ""]
        
        n_compare = min(length(current_errors), length(previous_errors))
        
        # Calculate mean manually: sum / length
        prev_mae = sum(previous_errors[1:n_compare]) / n_compare
        curr_mae = sum(current_errors[1:n_compare]) / n_compare

        diff = curr_mae - prev_mae
        pct_change = (diff / prev_mae) * 100

        if diff < 0
            println("✅ STATUS: IMPROVED! (Errors decreased by $(abs(diff)) | $(round(abs(pct_change), digits=2))%)")
        elseif diff > 0
            println("⚠️ STATUS: WORSENED! (Errors increased by $diff | +$(round(pct_change, digits=2))%)")
        else
            println("⚖️ STATUS: UNCHANGED. (Errors are identical)")
        end

        println("Previous Mean Error: $prev_mae")
        println("Current Mean Error:  $curr_mae")
        
        curr_max = maximum(current_errors[1:n_compare])
        prev_max = maximum(previous_errors[1:n_compare])
        println("Max Error Shift:     $prev_max -> $curr_max")
    else
        println("No previous run found. Saving current run as the baseline.")
        # Calculate mean manually here as well
        current_mean = sum(current_errors) / length(current_errors)
        println("Current Mean Error:  $current_mean")
    end

    # Write errors to a plain text file (one number per line)
    open(filename, "w") do file
        for err in current_errors
            println(file, err)
        end
    end
    
    println("Current run saved to '$filename' for future comparison.")
    println("==========================================\n")
end

params = SystemParams(
    ω1 = 1.0, 
    ω2 = 2.0, 
    ωp = 0, 
    ωq = 3.0, 
    g1 = 0.08, 
    g2 = 0.16,   
    g2p = 0, 
    θ = π / 6.0
)

results_full = get_optimal_frequency(H_full, params)
results_eff = get_optimal_frequency(H_eff, params)
results_num = get_optimal_frequency(H_num, params)

println("Optimal ω2 (full) = ", round(results_full[1], digits=6))
println("Optimal ω2 (eff) = ", round(results_eff[1], digits=6))
println("Optimal ω2 (num) = ", round(results_num[1], digits=6))

params_full = deepcopy(results_full[9])
params_eff = deepcopy(results_eff[9])
params_num = deepcopy(results_num[9])

Hfull = H_full(params_full)
Heff = H_eff(params_eff)
Hnum = H_num(params_num)

E_full     = real.(eigvals(Matrix(Hfull.data)))
E_eff      = real.(eigvals(Matrix(Heff.data)))
E_num      = real.(eigvals(Matrix(Hnum.data)))

sort!(E_full)
sort!(E_eff)
sort!(E_num)

num_levels = 30

# Shift relative to the ground state
E_full_shifted     = E_full[1:num_levels]     .- E_full[1]
E_eff_shifted      = E_eff[1:num_levels]      .- E_eff[1]
E_num_shifted      = E_num[1:num_levels]      .- E_num[1]

# Print a highly detailed 3-way comparison table
println("Level   | E_full          | E_eff (Analytical) | E_num (Numerical) | Err Analytical | Err Numerical")
println("-"^97)
for i in 1:num_levels
    err_eff = abs(E_full_shifted[i] - E_eff_shifted[i])
    err_num = abs(E_full_shifted[i] - E_num_shifted[i])
    
    # Use standard rounding and lpad (left-pad) to align columns
    c1 = lpad(i, 7)
    c2 = lpad(round(E_full_shifted[i], digits=6), 15)
    c3 = lpad(round(E_eff_shifted[i], digits=6), 18)
    c4 = lpad(round(E_num_shifted[i], digits=6), 17)
    
    # For errors, we use sigdigits (significant digits) to mimic scientific notation
    c5 = lpad(round(err_eff, sigdigits=3), 14)
    c6 = lpad(round(err_num, sigdigits=3), 13)
    
    println("$c1 | $c2 | $c3 | $c4 | $c5 | $c6")
end

# 7. Plotting
levels = 1:num_levels
errors_eff = abs.(E_full_shifted .- E_eff_shifted)
errors_num = abs.(E_full_shifted .- E_num_shifted)

# Monitor only the analytical errors for your regression tracker
MonitorRegression(errors_eff)

##

fig = Figure(size = (900, 700))

# Make x-ticks step by 2 so the axis doesn't get crowded with 30 numbers
x_ticks_display = 1:num_levels

ax1 = Axis(fig[1, 1], 
    title = "Eigenvalues Comparison",
    ylabel = L"Energy ($E - E_0$)",
    xticks = x_ticks_display
)

scatterlines!(ax1, levels, E_full_shifted, label="H_full", color=:blue, marker=:circle, markersize=12)
scatterlines!(ax1, levels, E_eff_shifted, label="H_eff", color=:darkorange, marker=:cross, markersize=10)
scatterlines!(ax1, levels, E_num_shifted, label="H_num", color=:purple, marker=:utriangle, markersize=10)
axislegend(ax1, position = :lt)

ax2 = Axis(fig[2, 1], 
    title = "Absolute Error with respect to H_full",
    xlabel = "Level Index", 
    ylabel = L"Error ($|E - E_{\text{full}}|$)",
    xticks = x_ticks_display
)
# Plot both error lines for direct visual comparison of the "bleeding" effect
scatterlines!(ax2, levels, errors_eff, label="Error: Analytical", color=:red, marker=:diamond, markersize=10)
scatterlines!(ax2, levels, errors_num, label="Error: Numerical", color=:purple, marker=:utriangle, markersize=10)
axislegend(ax2, position = :lt)

linkxaxes!(ax1, ax2)
display(fig)