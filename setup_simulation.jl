include("setup.jl")
include("setup_saving_loading.jl")
include("setup_post_processing.jl")
include("setup_resonance_finder.jl")
include("Adapt_setup.jl")



function prepare_simulation(params::SystemParams, H_fun, F, kp, k1 = 5e-6, find_resonance = true, matrix_form = Val(true))
    

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

    if is_ideal
        println("Step 1: Ideal model")
        H_drive_op = F *(a2 + a2')
    else 
        if is_effective_model
            println("Step 1: Effective or RWA model")
            S = SW_generator(params)
            field_op = L2_eff_4th_order(params, kp, is_3rd_order)
            F_drive = is_RWA ? F / 2.0 : F
            H_drive_op = H_drive_eff_4th_order(params, F_drive, is_3rd_order)
            if is_RWA
                println("Transforming to RWA frame Drive and Field")
                #field_op = L2_eff_num_RWA(field_op)
                H_drive_op = H_drive_num_RWA(H_drive_op)
                H += H_drive_op
            end
        else
            println("Step 1: Full model")
            field_op = 1im * sqrt(kp / params.ω2) * (a2 - a2')
            H_drive_op = 1im * F * (a2 - a2')
        end
    end


    k1 = 0
    field_intrinsic = 1im * sqrt(k1/params.ω1) *(a1-a1')

    if is_RWA
        println("Step 2: RWA coherent + lab-frame dressed dissipator")

        

        # --- dissipator: built from LAB H so bath rates use physical frequencies ---
        H_lab           = H_eff_4th_order(params)
        
        
        fields  = (field_op,)
        T_baths = (0.0,)
        
        e_d, v_d, L_lab = liouvillian_dressed_nonsecular( H_lab, fields, T_baths; matrix_form = Val(true))
        _,   _,  L_lab_c  = liouvillian_dressed_nonsecular(H_lab, fields, T_baths; matrix_form = Val(false))
        V_mat = Array(v_d.data)        # eigenbasis of H_lab; maps dressed -> bare

        # coherent part of H_lab IN ITS OWN dressed basis is just Diagonal(e_d)
        #H_lab_diag = QuantumObject(spdiagm(0 => ComplexF64.(e_d)), type=Operator(), dims=dims_sys)
        H_lab_diag = QuantumObject(V_mat' * Array(H_lab.data) * V_mat , type = Operator(), dims = dims_sys)
        D_dressed   = L_lab   - liouvillian(H_lab_diag; matrix_form = Val(true))
        D_dressed_c = L_lab_c - liouvillian(H_lab_diag; matrix_form = Val(false))
        
        # express the RWA coherent Hamiltonian in the SAME (lab-dressed) basis
        H_mat = V_mat' * Array(H.data) * V_mat
        H_d   = QuantumObject(droptol!(sparse(H_mat), 1e-12), type=Operator(), dims=dims_sys)

        L_cpu          = liouvillian(H_d; matrix_form = Val(true))  + D_dressed
        L_cpu_concrete = liouvillian(H_d; matrix_form = Val(false)) + D_dressed_c

        println("Transferring Liouvillian to GPU...")
        L_tot_gpu = Adapt.adapt(CUSPARSE.CuSparseMatrixCSR, L_cpu)

        # initial state, rotated into the lab-dressed basis
        psi0_bare    = tensor(fock(N1,0), fock(N2,0), basis(Np,0), basis(Nq,1))
        rho0_bare    = Array(ket2dm(psi0_bare).data)
        rho0_dressed = QuantumObject(V_mat' * rho0_bare * V_mat, type=Operator(), dims=dims_sys)

    else
        println("Step 2: Time dependent model (Full or Effective)")
        # 1. Use the dressed Liouvillian only for time-dependent, lab-frame Hamiltonians
        T_baths = (0.0,)
        fields = (field_op,) 
        
        println("Generating Liouvillian...")
        e_d, v_d, L_cpu = liouvillian_dressed_nonsecular(H, fields, T_baths; matrix_form = matrix_form)
        V_mat = Array(v_d.data)
        _, _, L_cpu_concrete = liouvillian_dressed_nonsecular(H, fields, T_baths; matrix_form = Val(false))
        
        println("Transferring Liouvillian to GPU...")
        L_gpu = Adapt.adapt(CUSPARSE.CuSparseMatrixCSR, L_cpu)
        
        H_drive_dressed_dense = V_mat' * Array(H_drive_op.data) * V_mat
        H_drive_dressed_sparse = droptol!(sparse(H_drive_dressed_dense), 1e-12)
        H_drive_dressed_qobj = QuantumObject(H_drive_dressed_sparse, type=Operator(), dims=dims_sys)
        
        L_drive_dressed_cpu = liouvillian(H_drive_dressed_qobj; matrix_form = matrix_form)
        L_drive_dressed_gpu = Adapt.adapt(CUSPARSE.CuSparseMatrixCSR, L_drive_dressed_cpu)
        drive_func(p, t) = cos(params.ωd * t)
        
        L_tot_gpu = (L_gpu, (L_drive_dressed_gpu, drive_func))
        
        rho0_dressed = ket2dm(fock(N1*N2*Np*Nq, 0; dims = dims_sys))           
    end

    # 3. Initial State Preparation
    
    rho0_dressed_gpu = cu(rho0_dressed)

    return L_cpu_concrete, L_tot_gpu, rho0_dressed_gpu, V_mat, params, is_RWA
end

function run_simulation(params::SystemParams, H_fun, filename, F, kp, tmax, t_selected, nframes, save_dir, k1 = 5e-6, find_resonance = true)

    matrix_form = Val(true)
    mkpath(save_dir)
    L_cpu_concrete, L_tot_gpu, rho0_dressed_gpu, V_mat, params, is_RWA = prepare_simulation(params::SystemParams, H_fun, F, kp, k1, find_resonance, matrix_form)

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