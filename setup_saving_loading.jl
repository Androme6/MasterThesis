using QuantumToolbox
using JLD2
using LinearAlgebra

function save_simulation(filepath, states_cpu_mats, V_mat, t, p::SystemParams, F, kp, tmax, nframes, expect_n1, expect_n2, expect_np)
    
    params_dict = Dict(
            "ω1" => p.ω1, "ω2" => p.ω2, "ωp" => p.ωp, "ωq" => p.ωq,
            "g1" => p.g1, "g2" => p.g2, "g1p" => p.g1p, "g2p" => p.g2p,
            "θ" => p.θ, "ωd" => p.ωd, "F" => F,
            "kp" => kp, "N1" => N1, "N2" => N2, "Np" => Np, "Nq" => Nq,
            "tmax" => tmax, "nframes" => nframes
        )

    jldsave(filepath; 
        states_cpu_mats = states_cpu_mats,
        V_mat = V_mat,
        t = t,
        params = params_dict,
        expect_n1 = expect_n1,
        expect_n2 = expect_n2,
        expect_np = expect_np
    )
    println("Minimal simulation data saved to: ", filepath)
end

function load_simulation(filepath)
    println("Loading data from: ", filepath)
    data = load(filepath)

    params = data["params"]
    V_mat = data["V_mat"]
    t = data["t"]
    states_cpu_mats = data["states_cpu_mats"]
    
    expect_n1 = data["expect_n1"]
    expect_n2 = data["expect_n2"]
    expect_np = data["expect_np"]
    
    return (
        states_cpu_mats = states_cpu_mats, 
        V_mat = V_mat, 
        t = t,
        expect_n1 = expect_n1, 
        expect_n2 = expect_n2, 
        expect_np = expect_np,
        params = params
    )
end