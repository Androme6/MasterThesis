include("setup.jl")
include("setup_saving_loading.jl")
include("setup_post_processing.jl")
include("setup_resonance_finder.jl")
include("Adapt_setup.jl")

function run_simulation(params::SystemParams, H_fun, filename, F, kp, tmax, t_selected, nframes, save_dir, k1 = 5e-6, find_resonance = true)

    mkpath(save_dir)
    matrix_form = Val(true)

    is_effective_model = (H_fun != H_full && H_fun != H_ideal)
    is_RWA = (H_fun == H_eff_RWA || H_fun == H_eff_num_RWA)
    is_3rd_order = (H_fun == H_eff_3rd_order)
    is_ideal = (H_fun == H_ideal)

    # 1. Find optimal frequencies
    if find_resonance && !is_ideal
            results = get_optimal_frequency(H_fun, params)
            println("Optimal ω2 = ", round(results[1], digits=6))
            println("ω2 dressed = ", round(results[2], digits=6))
            display(results[5])
            params = deepcopy(results[9])
            flush(stdout)
    end

    H = H_fun(params)

    # 2. Liouvillian & Drive Setup, and their Dressing
    field_pretransf_ext = 1im * sqrt(kp / params.ω2) * (a2_ext - a2_ext')
    F_drive = is_RWA ? F / 2.0 : F
    H_drive_pretransf_ext = 1im * F_drive * (a2_ext - a2_ext')

    if is_effective_model
        println("Applying SW transformation to drive and fields...")
        S = SW_generator(params)
        
        # Transform Field
        field_op = L2_eff_4th_order(params, kp, is_3rd_order)
    
        # Transform Drive
        H_drive_op = H_drive_eff_4th_order(params, F_drive, is_3rd_order)
    elseif is_ideal
        H_drive_op = F *(a2 + a2')   
    else
        println("Using bare drive and fields (H_full or RWA model)...")
        # Just project the bare operators down to target space
        field_final_mat = P_full_mat * field_pretransf_ext.data * P_full_mat'
        field_op = QuantumObject(field_final_mat, type=Operator(), dims=dims_sys)
        H_drive_op = QuantumObject(P_full_mat * H_drive_pretransf_ext.data * P_full_mat',  type=Operator(), dims=dims_sys)
    end


    if is_RWA
        println("Generating Standard Liouvillian on CPU (RWA uses bare baths)...")
    
        H_drive_op_RWA = H_drive_num_RWA(H_drive_op)
    
        # For the RWA model, the drive is static and added directly to H
        H_tot = H + H_drive_op_RWA

        #jump_a1, jump_a2 = L2_eff_num_RWA(field_op)
        jump_a1, jump_a2 = L2_eff_RWA(params, kp)
        k_eff_1, k_eff_2 = extract_effective_kappas(jump_a1, jump_a2)

        
        bare_a1_loss = sqrt(k1) * a1_ext 
        a1_loss_mat = P_full_mat * bare_a1_loss.data * P_full_mat'
        jump_a1_intrinsic = QuantumObject(a1_loss_mat, type=Operator(), dims=dims_sys)
        
        c_ops = [jump_a1, jump_a2, jump_a1_intrinsic]

        L_cpu = liouvillian(H_tot, c_ops; matrix_form = matrix_form)
        L_cpu_concrete = liouvillian(H_tot, c_ops; matrix_form = Val(false))
    
        println("Transferring Liouvillian to GPU...")
        L_tot_gpu = Adapt.adapt(CUSPARSE.CuSparseMatrixCSR, L_cpu)
    
        # The RWA model is already in the target bare basis, so V_mat is just the Identity matrix
        V_mat = Matrix{ComplexF64}(I, size(H_tot.data, 1), size(H_tot.data, 2))
    
        psi0_dressed = tensor(fock(N1, 0), fock(N2, 0), fock(Np, 0), fock(2, 1))

    elseif is_ideal
        H_tot = H + H_drive_op
        c_ops = [sqrt(kp) * a2, sqrt(k1) * a1]
        L_cpu = liouvillian(H_tot, c_ops; matrix_form = matrix_form)
        L_tot_gpu = Adapt.adapt(CUSPARSE.CuSparseMatrixCSR, L_cpu)
        L_cpu_concrete = liouvillian(H_tot, c_ops; matrix_form = Val(false))
        V_mat = Matrix{ComplexF64}(I, size(H_tot.data, 1), size(H_tot.data, 2))
        psi0_dressed = fock(N1*N2*Np*Nq, 0; dims = dims_sys)

    else
        println("Generating Dressed Liouvillian on CPU...")
        fields = (field_op,) 
        T_baths = (0.0,)
    
        e_d, v_d, L_cpu = liouvillian_dressed_nonsecular(H, fields, T_baths; matrix_form = matrix_form)
        V_mat = Array(v_d.data)
        _, _, L_cpu_concrete = liouvillian_dressed_nonsecular(H, fields, T_baths; matrix_form = Val(false))
    
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

    # 3. Initial State Preparation
    psi0_dressed_mat = ket2dm(psi0_dressed)
    psi0_dressed_gpu = cu(psi0_dressed_mat)


    # 4. Time Evolution
    println("Time evolution on GPU...")
    t = LinRange(0.0, tmax, nframes)

    sol_gpu = mesolve(L_tot_gpu, psi0_dressed_gpu, t, 
                  reltol=1e-5, abstol=1e-7,
                  maxiters=1e9, matrix_form = matrix_form)

    # 5. Extract raw states and move completely to CPU 
    println("Simulation complete. Moving states to CPU...")
    states_cpu_mats = [Array(state.data) for state in sol_gpu.states]

    # 6. Calculate occupations
    expect_n1, expect_n2, expect_np = calculate_occupations(states_cpu_mats, V_mat)

    # 7. Define paths and Save Data
    timestamp = Dates.format(now(), "yyyy-mm-dd_HHMMSS")
    filename = filename * "_" * timestamp
    save_path_data = joinpath(save_dir, filename * ".jld2")
    #save_simulation(save_path_data, states_cpu_mats, V_mat, t, params, F, kp, tmax, nframes, expect_n1, expect_n2, expect_np)

    # 8. Plotting and Exporting
    fig_master = analysis_and_plots(states_cpu_mats, V_mat, t, t_selected, params, expect_n1, expect_n2, expect_np, F, kp, N1, N2, Np, Nq, save_dir, filename, is_RWA)

    display(fig_master)

    return (
        states_cpu_mats = states_cpu_mats, 
        L_cpu = L_cpu_concrete,
        V_mat = V_mat, 
        t = t,
        expect_n1 = expect_n1, 
        expect_n2 = expect_n2, 
        expect_np = expect_np,
        params = params,
        fig_master = fig_master
    )
end