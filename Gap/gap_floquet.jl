include("../setup_simulation.jl")
include("setup_gap_floquet.jl")

params = SystemParams(
    ω1 = 5.0,  
    θ = π / 6.0,
    F = 0.5
)

F_list = range(0.05, 0.2, length=10)

pr = Progress(length(F_list))
res_td = map(F_list) do F

    params.F = F
    L_cpu, L_drive_dressed_cpu, _, _, V_mat, params_new, _ = prepare_simulation(params, H_fun, true, Val(false), false)

    drive_func(p, t) = cos(params_new.ωd * t)
    L_tot_cpu = (L_cpu, (L_drive_dressed_cpu, drive_func))

    a1_dressed = QuantumObject(V_mat' * Array(a1.data) * V_mat, type=Operator(), dims=dims_sys)
    res = gap_and_ss_population(L_tot_cpu, nothing, a1_dressed, params_new.ωd; params=[params_new.ωd], eigvals=4, eigstol=1e-8)

    next!(pr)
    res
end

ss_population_td = getindex.(res_td, 1)
gap_td = getindex.(res_td, 2)
