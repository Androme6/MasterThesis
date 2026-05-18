include("setup.jl")
include("setup_saving_loading.jl")
include("setup_post_processing.jl")
include("setup_resonance_finder.jl")
include("Adapt_setup.jl")

# 1. Set parameters 
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

F = 0.025
kp = 0.1 
tmax = 5#15000
t_selected = tmax
nframes = 5#500
H_fun = H_eff

save_dir = "C:\\Users\\andre\\Desktop\\Università\\Magistrale\\MA4\\Thesis\\Code\\MasterThesis\\Output"
filename = "Three_modes_full"
matrix_form = Val(true)

# 2. Find optimal frequencies
results = get_optimal_frequency(H_fun, params, ω2_list, ωp_list, lower_index_2, upper_index_2, lower_index_p, upper_index_p)
println("Optimal ω2 = ", round(results[1], digits=6))
println("ω2 dressed = ", round(results[2], digits=6))
display(results[5])
params = deepcopy(results[9])
ωd = results[2]

H = H_fun(params)

# 3. Jump Operators (now on Buffer)
#field = 1im * sqrt(kp / params.ωp) * (ap - ap')
field = 1im * sqrt(kp / params.ω2) * (a2 - a2') #for now
fields = (field,)
T_baths = (0.0,)

# 4. Dressed Liouvillian
println("Generating Dressed Liouvillian on CPU...")
e_d, v_d, L_cpu = liouvillian_dressed_nonsecular(H, fields, T_baths; matrix_form = matrix_form)
V_mat = Array(v_d.data)
println("Transferring Liouvillian to GPU...")
L_gpu = Adapt.adapt(CUSPARSE.CuSparseMatrixCSR, L_cpu)

# 5. Drive Hamiltonian (now on Buffer)
#H_drive_bare = 1im * F * (ap - ap')
H_drive_bare = 1im * F * (a2 - a2') #for now
H_drive_dressed_dense = V_mat' * Array(H_drive_bare.data) * V_mat
H_drive_dressed_sparse = droptol!(sparse(H_drive_dressed_dense), 1e-12)
H_drive_dressed_qobj = QuantumObject(H_drive_dressed_sparse, type=Operator(), dims=dims_sys)
L_drive_dressed_cpu = liouvillian(H_drive_dressed_qobj; matrix_form = matrix_form)
L_drive_dressed_gpu = Adapt.adapt(CUSPARSE.CuSparseMatrixCSR, L_drive_dressed_cpu)
drive_func(p, t) = cos(ωd * t)

# 6. Initial State Preparation
psi0_dressed = fock(N1*N2*Np*Nq, 0; dims = dims_sys) 
psi0_dressed_mat = ket2dm(psi0_dressed)
psi0_dressed_gpu = cu(psi0_dressed_mat)

# 7. Time Evolution
println("Time evolution on GPU...")
L_tot_gpu = (L_gpu, (L_drive_dressed_gpu, drive_func))

t = LinRange(0.0, tmax, nframes)

sol_gpu = mesolve(L_tot_gpu, psi0_dressed_gpu, t_save, 
                  reltol=1e-5, abstol=1e-7,
                  maxiters=1e9, matrix_form = matrix_form)

# 8. Extract raw states and move completely to CPU 
println("Simulation complete. Moving states to CPU...")
states_cpu_mats = [Array(state.data) for state in sol_gpu.states]

# 9. Calculate occupations
expect_n1, expect_n2, expect_np = calculate_occupations(states_cpu_mats, V_mat)

# 10. Define paths and Save Data
timestamp = Dates.format(now(), "yyyy-mm-dd_HHMMSS")
filename_run = filename * "_" * timestamp
mkpath(save_dir)

save_path_data = joinpath(save_dir, filename_run * ".jld2")
save_simulation(save_path_data, states_cpu_mats, V_mat, t, params, ωd, F, kp, tmax, nframes, expect_n1, expect_n2, expect_np)

# 11. Plotting and Exporting
fig_master = analysis_and_plots(states_cpu_mats, V_mat, t, params, expect_n1, expect_n2, expect_np, ωd, F, kp, save_dir, filename_run)
display(fig_master)