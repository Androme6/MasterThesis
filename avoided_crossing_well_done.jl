using QuantumToolbox
using Plots
using ProgressMeter
using LaTeXStrings

const N1 = 45
const N2 = 5
const Np = 1
const Nq = 2

const a1 = tensor(destroy(N1), qeye(N2), qeye(Np), qeye(Nq))
const a2 = tensor(qeye(N1), destroy(N2), qeye(Np), qeye(Nq))
const ap = tensor(qeye(N1), qeye(N2), destroy(Np), qeye(Nq))
const σz = tensor(qeye(N1), qeye(N2), qeye(Np), sigmaz())
const σy = tensor(qeye(N1), qeye(N2), qeye(Np), sigmay())
const σx = tensor(qeye(N1), qeye(N2), qeye(Np), sigmax())
const Id = tensor(qeye(N1), qeye(N2), qeye(Np), qeye(Nq))

##

# from here my code
@kwdef mutable struct SystemParams
    ω1::Float64 
    ω2::Float64 
    ωp::Float64
    ωq::Float64
    g1::Float64
    g2::Float64
    g2p::Float64
    g1p::Float64 = g2p * sqrt(ω1) / sqrt(ω2)
    θ::Float64
end


function H_full(p::SystemParams)
    H0 = p.ω1 * a1'*a1 + p.ω2 * a2'*a2 + p.ωp * ap'*ap + p.ωq * σz / 2
    Hint = (p.g1 * (a1+a1') + p.g2 * (a2+a2')) * (sin(p.θ) * σz + cos(p.θ) * σx)
    Hint_P = (p.g1p * (a1+a1') + p.g2p * (a2+a2')) * (ap+ap')
    return H0 + Hint + Hint_P
end

function H_eff(p::SystemParams)
    H0 = p.ω1 * a1'*a1 + p.ω2 * a2'*a2 + p.ωp * ap'*ap + p.ωq * σz / 2
    Hint_P = (p.g1p * (a1+a1') + p.g2p * (a2+a2')) * (ap+ap')
    
    anticomm(op1, op2) = op1 * op2 + op2 * op1

    g  = [p.g1, p.g2]
    gp = [p.g1p, p.g2p]
    ω  = [p.ω1, p.ω2]
    A  = [2*p.ω1/(p.ω1^2 - p.ωq^2), 2*p.ω2/(p.ω2^2 - p.ωq^2)]
    B  = [2*p.ωq/(p.ω1^2 - p.ωq^2), 2*p.ωq/(p.ω2^2 - p.ωq^2)]
    
    X  = [a1 + a1', a2 + a2']
    P  = [1im * (a1' - a1), 1im * (a2' - a2)]
    XP = ap + ap'
    
    sin_t  = sin(p.θ)
    cos_t  = cos(p.θ)
    sin_2t = sin(2*p.θ)
    

    H2       = 0.0 * Id
    H3       = 0.0 * Id
    H_filter = 0.0 * Id

    for i in 1:2
        # --- Filter (1st Order) ---
        term_z_f = -2.0 * sin_t * (g[i] * gp[i] / ω[i]) * XP * σz
        term_x_f = -cos_t * g[i] * gp[i] * A[i] * XP * σx
        H_filter += 0.5 * (term_z_f + term_x_f) 
        
        # --- 2nd Order: Single index terms ---
        H2 += -2 * sin_t^2 * (g[i]^2 / ω[i]) * Id           # [Sz, Vz]
        H2 += -cos_t^2 * g[i]^2 * A[i] * Id                 # [Sx, Vx] scalar part
        
        # --- 2nd Order: Double index terms ---
        for j in 1:2
            H2 += -cos_t^2 * g[i]*g[j] * B[i] * X[i]*X[j] * σz                # [Sx, Vx] op part
            H2 += (sin_2t / 2) * (g[i]*g[j] / ω[i]) * anticomm(P[i], X[j]) * σy # [Sz, Vx]
            
            # [Sx, Vz]
            term_A = -(A[i] / 2) * anticomm(P[i], X[j]) * σy
            term_B = B[i] * X[i]*X[j] * σx
            H2 += (sin_2t / 2) * g[i]*g[j] * (term_A + term_B)
        end
    end
    H2 *= 0.5 # Apply global 1/2 factor for 2nd order

    # --- 3rd Order: Double Sums ---
    for i in 1:2, k in 1:2
        # [Sz, [Sx, Vx]]
        H3 += 2 * sin_t * cos_t^2 * (g[i] * g[k]^2 / ω[k]) * (B[k] + B[i]) * X[i]
        # [Sz, [Sz, Vx]] (term 2)
        H3 += -sin_t * sin_2t * 2im * (g[i]^2 * g[k] / (ω[i]*ω[k])) * P[k] * σx

        # [Sx, [Sz, Vx]] (term 2)
        term2_SxSzVx = 2* (g[i]^2 * g[k] / ω[i]) * (B[i] * X[k] + 1im * A[k] * P[k] * σz)
        H3 += (cos_t * sin_2t / 2) * term2_SxSzVx
        
        # [Sx, [Sx, Vz]] (term 2)
        part_c = 2 * (2*B[i] + B[k]) * X[k]
        part_d = 2im * A[k] * P[k] * σz
        term2_SxSxVz = A[i] * g[i]^2 * g[k] * (part_c + part_d)
        H3 -= (cos_t * sin_2t / 4) * term2_SxSxVz
    end

    # --- 3rd Order: Triple Sums ---
    for i in 1:2, j in 1:2, k in 1:2
        # [Sz, [Sz, Vx]] (term 1)
        H3 += -sin_t * sin_2t * (g[i]*g[j]*g[k] / (ω[i]*ω[k])) * anticomm(P[k], P[i]*X[j]) * σx
        
        # [Sz, [Sx, Vz]]
        term_A_1 = A[i] * anticomm(P[k], P[i]*X[j]) * σx
        term_B_1 = B[i] * anticomm(P[k], X[i]*X[j]) * σy
        term_C_1 = 2im * A[i] * (i == j ? 1.0 : 0.0) * P[k] * σx
        H3 += sin_t * (sin_2t/2) * (g[i]*g[j]*g[k] / ω[k]) * (term_A_1 + term_B_1 + term_C_1)
        
        # [Sx, [Sx, Vx]]
        term_A_2 = A[k] * anticomm(P[k], X[i]*X[j]) * σy
        term_B_2 = -2 * B[k] * X[k]*X[i]*X[j] * σx
        H3 += (cos_t^3 / 2) * g[i]*g[j]*g[k] * B[i] * (term_A_2 + term_B_2)
        
        # [Sx, [Sz, Vx]] (term 1)
        H3 += (cos_t * sin_2t / 2) * (g[i]*g[j]*g[k] / ω[i]) * A[k] * anticomm(P[k], P[i]*X[j]) * σz
        
        # [Sx, [Sx, Vz]] (term 1)
        part_a = A[i] * A[k] * anticomm(P[k], P[i]*X[j]) * σz
        part_b = 2 * B[i] * B[k] * X[k]*X[i]*X[j] * σz
        H3 -= (cos_t * sin_2t / 4) * g[i]*g[j]*g[k] * (part_a + part_b)
    end
    H3 *= (1.0 / 3.0) # Apply global 1/3 factor for 3rd order
    
    return H0 + Hint_P + H2 + H3 + H_filter
end

function resonance_finder(H_fun, p, ω2_list, ωp_list, lower_index_2, upper_index_2, lower_index_p, upper_index_p)
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
    fig2 = Plots.plot(ylims=(1.9, 2.05),
        xlabel=L"\omega_2/\omega_1", ylabel=L"E/\hbar\omega_1",
        legend=:bottomright, size=(800, 600), framestyle=:box)
    for i in 1:size(eigenvalues, 1)
        Plots.plot!(fig2, ω2_list ./ p.ω1, real(eigenvalues[i, :] - eigenvalues[1,:]))
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
            H = H_full(p)
            eigenstates(H, sparse = true, sigma = -p.ωq, eigvals = 7).values
    end
    #plotting
    figp = Plots.plot(ylims=(1.9, 2.05),
    xlabel=L"\omega_p/\omega_1", ylabel=L"E/\hbar\omega_1",
    legend=:bottomright, size=(800, 600), framestyle=:box)
    for i in 1:size(eigenvalues_vs_p, 1)
        Plots.plot!(figp, ωp_list ./ p.ω1, real(eigenvalues_vs_p[i, :] - eigenvalues_vs_p[1,:]))
    end
    #finding optimal point
    gap_p, idx_opt_p = findmin(real, eigenvalues_vs_p[upper_index_p, :] - eigenvalues_vs_p[lower_index_p, :])
    ωp_opt = ωp_list[idx_opt_p]
    ωp_dressed = real(eigenvalues_vs_p[upper_index_p, 1] - eigenvalues_vs_p[1, 1])

    return ω2_opt, ω2_dressed, ωp_opt, ωp_dressed, fig2, figp, gap, gap_p
end


##
params = SystemParams(
    ω1 = 1.0, 
    ω2 = 2.0, 
    ωp = 0.0, #2.0, 
    ωq = 2.5, 
    g1 = 0.1, 
    g2 = 0.2,   
    g2p = 0, #0.02,
    θ = π / 6.0
)
ω2_list = range(1.9 * params.ω1, 2.1 * params.ω1, length=300)
ωp_list = range(1.7 * params.ω1, 2.2 * params.ω1, length=300)
lower_index_2 = 3
upper_index_2 = 4
lower_index_p = 3
upper_index_p = 5

results = resonance_finder(H_full, params, ω2_list, ωp_list, lower_index_2, upper_index_2, lower_index_p, upper_index_p)
println("Optimal ω2 = ", round(results[1], digits=6))
println("ω2 dressed = ", round(results[2], digits=6))
println("Optimal ωp = ", round(results[3], digits=6))
println("ωp dressed = ", round(results[4], digits=6))
println("Gap at optimal ω2 = ", round(results[7], digits=6))
println("Gap at optimal ωp = ", round(results[8], digits=6))
display(results[5])  # Plot for ω2 sweep
display(results[6])  # Plot for ωp sweep

results_eff = resonance_finder(H_eff, params, ω2_list, ωp_list, lower_index_2, upper_index_2, lower_index_p, upper_index_p)
println("Optimal ω2 = ", round(results_eff[1], digits=6))
println("ω2 dressed = ", round(results_eff[2], digits=6))
println("Optimal ωp = ", round(results_eff[3], digits=6))
println("ωp dressed = ", round(results_eff[4], digits=6))
println("Gap at optimal ω2 = ", round(results_eff[7], digits=6))
println("Gap at optimal ωp = ", round(results_eff[8], digits=6))
display(results_eff[5])  # Plot for ω2 sweep
display(results_eff[6])  # Plot for ωp sweep