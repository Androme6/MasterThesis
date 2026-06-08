include("setup.jl")
include("setup_saving_loading.jl")
include("setup_post_processing.jl")
include("setup_resonance_finder.jl")
include("Adapt_setup.jl")



function prepare_simulation(params::SystemParams, H_fun, F, kp, k1 = 5e-6, find_resonance = true, matrix_form = Val(true), numerical_RWA = false)
    

    is_effective_model = (H_fun != H_full && H_fun != H_ideal)
    is_RWA_qubit = (H_fun == H_eff_RWA || H_fun == H_RWA_qubit )
    is_RWA = (H_fun == H_RWA) 
    is_qubit = ( H_fun == H_qubit)
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


    # 2. Build H_drive_op and field_op
    if is_ideal
        println("Step 1: Ideal model")
        H_drive_op = F *(a2 + a2')
        H += H_drive_op
    elseif is_effective_model
        println("Step 1: Effective or RWA or qubit or RWA-qubit model")
        S = SW_generator(params)
        field_op = L2_eff_4th_order(params, kp, is_3rd_order)
        F_drive = (is_RWA || is_RWA_qubit || numerical_RWA) ? F / 2.0 : F
        H_drive_op = H_drive_eff_4th_order(params, F_drive, is_3rd_order)
        if is_RWA_qubit
            println("RWA-qubit model")
            H_drive_op = H_drive_RWA_qubit(H_drive_op)
        elseif is_qubit
            println("Qubit model")
            H_drive_op = H_drive_qubit(H_drive_op)
        elseif is_RWA
            println("RWA model")
            H_drive_op = H_drive_RWA(H_drive_op)
        end

        if numerical_RWA
            println("Numerical RWA model")
            H_drive_op = perform_numerical_RWA(H, H_drive_op, params, kp)
        end
    else
        println("Step 1: Full model")
        field_op = 1im * sqrt(kp / params.ω2) * (a2 - a2')
        H_drive_op = 1im * F * (a2 - a2')
    end



    # 3. Build Liouvillian and initial state
    if is_ideal
        println("Step 2: Ideal model")
        c_ops = [sqrt(kp) * a2, sqrt(k1) * a1]
        println("Generating Liouvillian...")
        L_cpu = liouvillian(H, c_ops; matrix_form = matrix_form)
        V_mat = Matrix{ComplexF64}(I, size(H.data, 1), size(H.data, 2))
        L_cpu_concrete = liouvillian(H, c_ops; matrix_form = Val(false))
        println("Transferring Liouvillian to GPU...")
        L_tot_gpu = Adapt.adapt(CUSPARSE.CuSparseMatrixCSR, L_cpu)
        psi0_dressed = fock(N1*N2*Np*Nq, 0; dims = dims_sys)
        rho0_dressed_gpu = cu(ket2dm(psi0_dressed))
    else
        println("Step 2: Full or Effective or RWA or qubit or RWA-qubit model")
        T_baths = (0.0,)
        fields = (field_op,) 
        println("Generating Liouvillian...")
        _ , v_d, L_cpu = liouvillian_dressed_nonsecular(H, fields, T_baths; matrix_form = matrix_form)
        V_mat = Array(v_d.data)
        _, _, L_cpu_concrete = liouvillian_dressed_nonsecular(H, fields, T_baths; matrix_form = Val(false))
        println("Transferring Liouvillian to GPU...")
        L_gpu = Adapt.adapt(CUSPARSE.CuSparseMatrixCSR, L_cpu)

        H_drive_dressed_dense =  V_mat' * Array(H_drive_op.data) * V_mat
        H_drive_dressed_sparse = droptol!(sparse(H_drive_dressed_dense), 1e-12)
        H_drive_dressed_qobj = QuantumObject(H_drive_dressed_sparse, type=Operator(), dims=dims_sys)
        L_drive_dressed_cpu = liouvillian(H_drive_dressed_qobj; matrix_form = matrix_form)
        L_drive_dressed_gpu = Adapt.adapt(CUSPARSE.CuSparseMatrixCSR, L_drive_dressed_cpu)
        
        rho0_dressed_gpu = cu(ket2dm(fock(N1*N2*Np*Nq, 0; dims = dims_sys)))
        #rho0_dressed_gpu = cu(QuantumObject(V_mat' * Array(ket2dm(tensor(fock(N1, 0), fock(N2, 0), fock(Np, 0), fock(Nq, 0))).data)* V_mat, type=Operator(), dims=dims_sys))
            

        if is_RWA || is_RWA_qubit || numerical_RWA
            println("Step 2: Time independent model (RWA or RWA-qubit or Numerical RWA)")
            L_tot_gpu = L_gpu + L_drive_dressed_gpu
        else
            println("Step 2: Time dependent model (Full or Effective or qubit)")
            drive_func(p, t) = cos(params.ωd * t)
            L_tot_gpu = (L_gpu, (L_drive_dressed_gpu, drive_func))
        end
    end

    return L_cpu_concrete, L_tot_gpu, rho0_dressed_gpu, V_mat, params, is_RWA, is_RWA_qubit
end

function run_simulation(params::SystemParams, H_fun, filename, F, kp, tmax, t_selected, nframes, save_dir, k1 = 5e-6, find_resonance = true, numerical_RWA = false)

    matrix_form = Val(true)
    mkpath(save_dir)
    L_cpu_concrete, L_tot_gpu, rho0_dressed_gpu, V_mat, params, is_RWA, is_RWA_qubit = prepare_simulation(params::SystemParams, H_fun, F, kp, k1, find_resonance, matrix_form, numerical_RWA)

    # 4. Time Evolution
    println("Time evolution on GPU...")
    t = LinRange(0.0, tmax, nframes)

    sol_gpu = mesolve(L_tot_gpu, rho0_dressed_gpu, t, 
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
    fig_master = analysis_and_plots(states_cpu_mats, V_mat, t, t_selected, params, expect_n1, expect_n2, expect_np, F, kp, N1, N2, Np, Nq, save_dir, filename, is_RWA, is_RWA_qubit, numerical_RWA)

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