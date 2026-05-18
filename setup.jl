using QuantumToolbox
using CairoMakie
using LaTeXStrings
using CUDA 
using SparseArrays
using Dates
using JLD2
using LinearAlgebra

const N1 = 45
const N2 = 5
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