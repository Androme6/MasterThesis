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





# Funzione per generare Jump Operators secolari e corretti per RWA
function generate_secular_jump_operators(params, field_eff::QuantumObject, J_omega::Function; secular_tol=1e-4)
    # 1. Diagonalizza l'Hamiltoniana nuda (nel LAB FRAME) per trovare le vere energie fisiche

    H_lab_0 = H_eff_4th_order(params)
    E, V_kets = eigenstates(H_lab_0)
    N = length(E)
    
    # Estraiamo la matrice del cambio di base (le colonne sono gli autovettori)
    V_mat = hcat([ket.data for ket in V_kets]...)
    
    # 2. Trasformiamo l'operatore di campo effettivo nella base delle energie
    # f_eig è la matrice con gli elementi <i | f_eff | j>
    f_matrix = field_eff.data
    f_eig = V_mat' * f_matrix * V_mat
    
    # Dizionario per raggruppare i jump operators per frequenza (Approssimazione Secolare)
    jump_ops_dict = Dict{Float64, Matrix{ComplexF64}}()
    
    # 3. Analizza ogni singola transizione possibile
    for j in 1:N       # Indice dello stato iniziale (energia più alta)
        for i in 1:N   # Indice dello stato finale (energia più bassa)
            
            # Frequenza della transizione nel lab frame
            omega = real(E[j] - E[i]) 
            
            # Se omega > 0, il sistema perde energia (salto di decadimento fisico a T=0)
            if omega > 1e-8 
                
                # Elemento di matrice della transizione
                element = f_eig[i, j]
                
                if abs(element) > 1e-12 # Ignora accoppiamenti nulli
                    
                    # 4. Applica il fattore della densità spettrale del bagno J(ω)
                    rate_amplitude = sqrt(J_omega(omega))
                    
                    # Costruisci l'operatore di salto |i><j| tornando alla base originale
                    # V_mat[:, i] è il ket |i>, V_mat[:, j]' è il bra <j|
                    op_ij_orig = rate_amplitude * element * (V_mat[:, i] * V_mat[:, j]')
                    
                    # 5. Raggruppa per frequenza (Approssimazione Secolare)
                    # Arrotonda la frequenza in base alla tolleranza per unire livelli degeneri
                    omega_rounded = round(omega / secular_tol) * secular_tol
                    
                    if haskey(jump_ops_dict, omega_rounded)
                        jump_ops_dict[omega_rounded] += op_ij_orig
                    else
                        jump_ops_dict[omega_rounded] = op_ij_orig
                    end
                end
            end
        end
    end
    
    # 6. Riconverti i risultati in QuantumObjects sparsi, pronti per il mesolve
    dims_sys = H_lab_0.dims
    c_ops_RWA = QuantumObject[]
    
    for mat in values(jump_ops_dict)
        mat_sparse = droptol!(sparse(mat), 1e-12)
        push!(c_ops_RWA, QuantumObject(mat_sparse, type=Operator(), dims=dims_sys))
    end
    
    return c_ops_RWA
end