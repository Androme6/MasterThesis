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
    θ = π / 6.0,
    ωd = 0.0
)

F = 0.025
kp = 0.1 
tmax = 15000
t_selected = tmax
nframes = 500

###########
H_fun = H_eff_num_RWA
filename = "Three_modes_RWA"
###########

save_dir = "C:\\Users\\andre\\Desktop\\Università\\Magistrale\\MA4\\Thesis\\Code\\MasterThesis\\Output"
#save_dir = "/capstor/store/cscs/2go/go072/alanteri"
mkpath(save_dir)

matrix_form = Val(true)

# 2. Find optimal frequencies
results = get_optimal_frequency(H_fun, params)
println("Optimal ω2 = ", round(results[1], digits=6))
println("ω2 dressed = ", round(results[2], digits=6))
display(results[5])
params = deepcopy(results[9])

flush(stdout)

H = H_fun(params)

H_test = H_eff_num_RWA(params)
difference_norm = norm(H.data - H_test.data)

is_effective_model = (H_fun != H_full)
is_RWA = (H_fun == H_eff_RWA || H_fun == H_eff_num_RWA)

# 3, 4, 5. Liouvillian & Drive Setup, and their Dressing
field_pretransf_ext = 1im * sqrt(kp / params.ω2) * (a2_ext - a2_ext')
F_drive = is_RWA ? F / 2.0 : F
H_drive_pretransf_ext = 1im * F_drive * (a2_ext - a2_ext')

if is_effective_model
    println("Applying SW transformation to drive and fields...")
    S = SW_generator(params)
    
    
    # Transform Field
    field_op = L2_eff_4th_order(params, kp)
    
    # Transform Drive
    d1 = commutator(S, H_drive_pretransf_ext)
    d2 = commutator(S, d1)
    d3 = commutator(S, d2)
    d4 = commutator(S, d3)
    H_drive_trans_ext = H_drive_pretransf_ext + d1 + (1.0/2.0)*d2 + (1.0/6.0)*d3 + (1.0/24.0)*d4
    
    # project down to target space
    #field_final_mat = P_full_mat * field_trans_ext.data * P_full_mat'
    H_drive_final_mat = P_full_mat * H_drive_trans_ext.data * P_full_mat'
else
    println("Using bare drive and fields (H_full or RWA model)...")
    # Just project the bare operators down to target space
    field_final_mat = P_full_mat * field_pretransf_ext.data * P_full_mat'
    field_op = QuantumObject(field_final_mat, type=Operator(), dims=dims_sys)
    H_drive_final_mat = P_full_mat * H_drive_pretransf_ext.data * P_full_mat'
end

H_drive_op = QuantumObject(H_drive_final_mat, type=Operator(), dims=dims_sys)

if is_RWA
    println("Generating Standard Liouvillian on CPU (RWA uses bare baths)...")
    
    H_drive_op_RWA = H_drive_num_RWA(H_drive_op)
    
    # For the RWA model, the drive is static and added directly to H
    H_tot = H + H_drive_op_RWA

    #jump_a1, jump_a2 = L2_eff_num_RWA(field_op)
    jump_a1, jump_a2 = L2_eff_RWA(params, kp)
    k_eff_1, k_eff_2 = extract_effective_kappas(jump_a1, jump_a2)
    c_ops = [jump_a1, jump_a2]

    L_cpu = liouvillian(H_tot, c_ops; matrix_form = matrix_form)
    
    println("Transferring Liouvillian to GPU...")
    L_tot_gpu = Adapt.adapt(CUSPARSE.CuSparseMatrixCSR, L_cpu)
    
    # The RWA model is already in the target bare basis, so V_mat is just the Identity matrix
    V_mat = Matrix{ComplexF64}(I, size(H_tot.data, 1), size(H_tot.data, 2))
    
    psi0_dressed = tensor(fock(N1, 0), fock(N2, 0), fock(Np, 0), fock(2, 1))

else
    println("Generating Dressed Liouvillian on CPU...")
    fields = (field_op,) 
    T_baths = (0.0,)
    
    e_d, v_d, L_cpu = liouvillian_dressed_nonsecular(H, fields, T_baths; matrix_form = matrix_form)
    V_mat = Array(v_d.data)
    
    println("Transferring Liouvillian to GPU...")
    L_gpu = Adapt.adapt(CUSPARSE.CuSparseMatrixCSR, L_cpu)
    
    # Drive Hamiltonian (now on Buffer)
    H_drive_dressed_dense = V_mat' * Array(H_drive_op.data) * V_mat
    H_drive_dressed_sparse = droptol!(sparse(H_drive_dressed_dense), 1e-12)
    H_drive_dressed_qobj = QuantumObject(H_drive_dressed_sparse, type=Operator(), dims=dims_sys)
    L_drive_dressed_cpu = liouvillian(H_drive_dressed_qobj; matrix_form = matrix_form)
    L_drive_dressed_gpu = Adapt.adapt(CUSPARSE.CuSparseMatrixCSR, L_drive_dressed_cpu)
    
    drive_func(p, t) = cos(params.ωd * t)
    L_tot_gpu = (L_gpu, (L_drive_dressed_gpu, drive_func))

    psi0_dressed = fock(N1*N2*Np*Nq, 0; dims = dims_sys) 
end

# 6. Initial State Preparation
psi0_dressed_mat = ket2dm(psi0_dressed)
psi0_dressed_gpu = cu(psi0_dressed_mat)


# 7. Time Evolution
println("Time evolution on GPU...")
t = LinRange(0.0, tmax, nframes)

sol_gpu = mesolve(L_tot_gpu, psi0_dressed_gpu, t, 
                  reltol=1e-5, abstol=1e-7,
                  maxiters=1e9, matrix_form = matrix_form)

# 8. Extract raw states and move completely to CPU 
println("Simulation complete. Moving states to CPU...")
states_cpu_mats = [Array(state.data) for state in sol_gpu.states]

# 9. Calculate occupations
expect_n1, expect_n2, expect_np = calculate_occupations(states_cpu_mats, V_mat)

# 10. Define paths and Save Data
timestamp = Dates.format(now(), "yyyy-mm-dd_HHMMSS")
filename = filename * "_" * timestamp
save_path_data = joinpath(save_dir, filename * ".jld2")
#save_simulation(save_path_data, states_cpu_mats, V_mat, t, params, F, kp, tmax, nframes, expect_n1, expect_n2, expect_np)

# 11. Plotting and Exporting
fig_master = analysis_and_plots(states_cpu_mats, V_mat, t, t_selected, params, expect_n1, expect_n2, expect_np, F, kp, save_dir, filename, is_RWA)