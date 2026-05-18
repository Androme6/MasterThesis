using QuantumToolbox
using CairoMakie
using LaTeXStrings
using CUDA 
using SparseArrays
using Dates
using JLD2
using LinearAlgebra

# 1. Define system parameters and operators
const N1 = 2
const N2 = 2
const Np = 1
const Nq = 2
const dims_sys = (N1, N2, Np, Nq)

const a1 = tensor(destroy(N1), qeye(N2), qeye(Np), qeye(Nq))
const a2 = tensor(qeye(N1), destroy(N2), qeye(Np), qeye(Nq))
const ap = tensor(qeye(N1), qeye(N2), destroy(Np), qeye(Nq))
const n1 = tensor(num(N1), qeye(N2), qeye(Np), qeye(Nq))
const n2 = tensor(qeye(N1), num(N2), qeye(Np), qeye(Nq))
const np = tensor(qeye(N1), qeye(N2), num(Np), qeye(Nq))
const σz = tensor(qeye(N1), qeye(N2), qeye(Np), sigmaz())
const σy = tensor(qeye(N1), qeye(N2), qeye(Np), sigmay())
const σx = tensor(qeye(N1), qeye(N2), qeye(Np), sigmax())
const Id = tensor(qeye(N1), qeye(N2), qeye(Np), qeye(Nq))

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



function save_simulation(filepath, sol_gpu, V_mat, t, p::SystemParams, ωd, F, kp, tmax, nframes)
    println("Preparing minimal data for saving (moving CuArrays to CPU)...")
    
    states_cpu_mats = [Array(state.data) for state in sol_gpu.states]
    
    params_dict = Dict(
            "ω1" => p.ω1, "ω2" => p.ω2, "ωp" => p.ωp, "ωq" => p.ωq,
            "g1" => p.g1, "g2" => p.g2, "g1p" => p.g1p, "g2p" => p.g2p,
            "θ" => p.θ, "ωd" => ωd, "F" => F,
            "kp" => kp, "N1" => N1, "N2" => N2, "Np" => Np, "Nq" => Nq,
            "tmax" => tmax, "nframes" => nframes
        )

    jldsave(filepath; 
        states_cpu_mats = states_cpu_mats,
        V_mat = V_mat,
        t = t,
        params = params_dict
    )
    println("Minimal simulation data saved to: ", filepath)
end
function post_processing(sol_gpu, V_mat, t_save, t_selected, ωd)
    println("Extracting memory counter-rotated density matrix...")
    t_selected_idx = argmin(abs.(t_save .- t_selected))
    rho_final_dressed_mat = Array(sol_gpu.states[t_selected_idx].data)
    rho_final_bare_mat = V_mat * rho_final_dressed_mat * V_mat'
    t_final = t_save[t_selected_idx]
    U_rot = exp(1im * ωd/2 * t_final * Array(n1.data))
    rho_final_bare_rotated_mat = U_rot * rho_final_bare_mat * U_rot'
    rho_final_bare_rotated_qobj = QuantumObject(rho_final_bare_rotated_mat, type=Operator(), dims=dims_sys)
    rho_mode1_rotated = ptrace(rho_final_bare_rotated_qobj, 1)
    return rho_mode1_rotated, t_selected_idx
end
function calculate_occupations(sol_gpu, V_mat, t_save)
    println("Calculating occupations...")
    expect_n1_plot = zeros(length(t_save))
    expect_n2_plot = zeros(length(t_save))
    expect_np_plot = zeros(length(t_save))

    n1_dressed_gpu_mat = cu(V_mat' * Array(n1.data) * V_mat)
    n2_dressed_gpu_mat = cu(V_mat' * Array(n2.data) * V_mat)
    np_dressed_gpu_mat = cu(V_mat' * Array(np.data) * V_mat)

    for i in 1:length(sol_gpu.states)
        rho_gpu = sol_gpu.states[i].data
        expect_n1_plot[i] = real(dot(n1_dressed_gpu_mat', rho_gpu))
        expect_n2_plot[i] = real(dot(n2_dressed_gpu_mat', rho_gpu))
        expect_np_plot[i] = real(dot(np_dressed_gpu_mat', rho_gpu))
    end
    return expect_n1_plot, expect_n2_plot, expect_np_plot
end


# 2. Define Hamiltonians
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



# 3. Set parameters and generate Hamiltonian
params = SystemParams(
    ω1 = 1.0, 
    ω2 = 2.048495, 
    ωp = 0.0, #for now, otherwise 1.985953
    ωq = 2.5, 
    g1 = 0.1, 
    g2 = 0.2,   
    g2p = 0.0, #for now, otherwise 0.02
    θ = π / 6.0
)
F = 0.065
kp = 0.13 
ωd = 1.985096

filename = "Three_modes_full"
tmax = 5#15000
nframes = 5#5000

H = H_full(params)

# 4. Jump Operators (now on Buffer)
#field = 1im * sqrt(kp / params.ωp) * (ap - ap')
field = 1im * sqrt(kp / params.ω2) * (a2 - a2') #for now
fields = (field,)
T_baths = (0.0,)

# 5. Dressed Liouvillian
println("Generating Dressed Liouvillian on CPU...")
e_d, v_d, L_cpu = liouvillian_dressed_nonsecular(H, fields, T_baths; matrix_form = Val(true))
V_mat = Array(v_d.data)
println("Transferring Liouvillian to GPU...")
L_gpu = cu(L_cpu)

# 6. Drive Hamiltonian (now on Buffer)
#H_drive_bare = 1im * F * (ap - ap')
H_drive_bare = 1im * F * (a2 - a2') #for now
H_drive_dressed_dense = V_mat' * Array(H_drive_bare.data) * V_mat
H_drive_dressed_sparse = droptol!(sparse(H_drive_dressed_dense), 1e-12)
H_drive_dressed_qobj = QuantumObject(H_drive_dressed_sparse, type=Operator(), dims=dims_sys)
L_drive_dressed_cpu = liouvillian(H_drive_dressed_qobj; matrix_form = Val(true))
L_drive_dressed_gpu = cu(L_drive_dressed_cpu)
drive_func(p, t) = cos(ωd * t)

# *7. Initial State Preparation
psi0_dressed = fock(N1*N2*Np*Nq, 0; dims = dims_sys) 
psi0_dressed_mat = ket2dm(psi0_dressed)
psi0_dressed_gpu = cu(psi0_dressed_mat)

# 8. Time Evolution
println("Time evolution on GPU...")
# L_tot_gpu = (L_gpu, (L_drive_dressed_gpu, drive_func))
L_tot_gpu = L_gpu + QobjEvo(L_drive_dressed_gpu, drive_func)

t_save = LinRange(0.0, tmax, nframes)

sol_gpu = mesolve(L_tot_gpu, psi0_dressed_gpu, t_save, 
                  reltol=1e-5, abstol=1e-7,
                  maxiters=1e9; matrix_form = Val(true))


# 9. Saving
timestamp = Dates.format(now(), "yyyy-mm-dd_HHMMSS")
filename = filename * "_" * timestamp
save_dir = joinpath(homedir(), "Thesis", "Output_New")
if !isdir(save_dir)
    mkpath(save_dir)
end
save_path_data = joinpath(save_dir, filename * ".jld2")
save_simulation(save_path_data, sol_gpu, V_mat, t_save, params, ωd, F, kp, tmax, nframes)

# 10. Wigner function
rho_mode1_rotated, t_selected_idx = post_processing(sol_gpu, V_mat, t_save, tmax, ωd)
xvec = LinRange(-5, 5, 100)
yvec = LinRange(-5, 5, 100)
# Wigner function of the counter-rotated state in first mode
W_cat = wigner(rho_mode1_rotated, xvec, yvec)'

# 11. Occupations
expect_n1_plot, expect_n2_plot, expect_np_plot = calculate_occupations(sol_gpu, V_mat, t_save)

# 12. Plotting
fig_master = Figure(size = (1200, 1000))

# Panel 1: Population Dynamics (Now includes Purcell Filter)
ax_pop = Axis(fig_master[1, 1], title="Occupation Numbers", xlabel=L"\text{Time} (1/ω_1)", ylabel="Average Occupation Number")
lines!(ax_pop, t_save, expect_np_plot, label="<nP> (Purcell Filter)", linewidth=3, color=:green, linestyle=:dash)
lines!(ax_pop, t_save, expect_n2_plot, label="<n2> (Buffer Mode)", linewidth=3, color=:orange)
lines!(ax_pop, t_save, expect_n1_plot, label="<n1> (Memory Mode)", linewidth=3, color=:blue)
axislegend(ax_pop, position=:lt)

# Panel 2: Fock State Populations Histogram (Top Right)
fock_populations = real.(diag(rho_mode1_rotated.data))
photon_numbers = 0:(N1-1)
ax_fock = Axis(fig_master[1, 2], 
               title="Fock State Populations (Memory Mode), t = $(round(t_save[t_selected_idx], digits=2))", 
               xlabel="Photon Number (n)", 
               ylabel="Probability P(n)")
barplot!(ax_fock, photon_numbers, fock_populations, color=:dodgerblue, strokecolor=:black, strokewidth=1)
CairoMakie.xlims!(ax_fock, -0.5, 30)

# Panel 3: 2D Wigner
ax2D = Axis(fig_master[2, 1], title = "2D Wigner Function (Memory Mode), t = $(round(t_save[t_selected_idx], digits=2))", xlabel = "Re(α)", ylabel = "Im(α)", aspect = 1) 
hm = CairoMakie.heatmap!(ax2D, xvec, yvec, W_cat, colormap = :RdBu)
Colorbar(fig_master[2, 1, Right()], hm, label = "W(α)")

# Panel 4: 3D Wigner
ax3D = Axis3(fig_master[2, 2], title = "3D Wigner Function (Memory Mode), t = $(round(t_save[t_selected_idx], digits=2))", xlabel = "Re(α)", ylabel = "Im(α)", zlabel = "W(α)", elevation = pi/6, azimuth = pi/4) 
CairoMakie.surface!(ax3D, xvec, yvec, W_cat, colormap = :RdBu)
CairoMakie.colgap!(fig_master.layout, 150)

save_path_img = joinpath(save_dir, filename * ".png")
CairoMakie.save(save_path_img, fig_master, px_per_unit = 2) 
println("Saved image to: ", save_path_img)

# 13. Summary Text
summary_text = """
--- System Parameters ---
ω1 = $(params.ω1) | ω2 = $(params.ω2) | ωp = $(params.ωp) | ωq  = $(params.ωq)
g1 = $(params.g1) | g2 = $(params.g2) | g1p = $(params.g1p) | g2p = $(params.g2p)
θ    = $(round(params.θ, digits=3))
κp  = $(kp)
F = $(F)
ωd  = $(round(ωd, digits=5))
N1 = $(N1) | N2 = $(N2) | Np = $(Np) | Nq = $(Nq)

Observables:
Final ⟨n1⟩ = $(round(expect_n1_plot[end], digits=4))
Final ⟨n2⟩ = $(round(expect_n2_plot[end], digits=4))
Final ⟨nP⟩ = $(round(expect_np_plot[end], digits=4))
"""
save_path_txt = joinpath(save_dir, filename * ".txt")
open(save_path_txt, "w") do file
    write(file, summary_text)
end
println("Saved logs to:  ", save_path_txt)


