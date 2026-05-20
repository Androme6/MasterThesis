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
    N1_ext = N1 + 1
    N2_ext = N2 + 1
    
    a1_ext = tensor(destroy(N1_ext), qeye(N2_ext), qeye(Np), qeye(Nq))
    a2_ext = tensor(qeye(N1_ext), destroy(N2_ext), qeye(Np), qeye(Nq))
    ap_ext = tensor(qeye(N1_ext), qeye(N2_ext), destroy(Np), qeye(Nq))
    σz_ext = tensor(qeye(N1_ext), qeye(N2_ext), qeye(Np), sigmaz())
    σy_ext = tensor(qeye(N1_ext), qeye(N2_ext), qeye(Np), sigmay())
    σx_ext = tensor(qeye(N1_ext), qeye(N2_ext), qeye(Np), sigmax())
    Id_ext = tensor(qeye(N1_ext), qeye(N2_ext), qeye(Np), qeye(Nq))

    H0 = p.ω1 * a1_ext'*a1_ext + p.ω2 * a2_ext'*a2_ext + p.ωp * ap_ext'*ap_ext + p.ωq * σz_ext / 2
    Hint_P = (p.g1p * (a1_ext+a1_ext') + p.g2p * (a2_ext+a2_ext')) * (ap_ext+ap_ext')
    
    g  = [p.g1, p.g2]
    gp = [p.g1p, p.g2p]
    ω  = [p.ω1, p.ω2]
    A  = [2*p.ω1/(p.ω1^2 - p.ωq^2), 2*p.ω2/(p.ω2^2 - p.ωq^2)]
    B  = [2*p.ωq/(p.ω1^2 - p.ωq^2), 2*p.ωq/(p.ω2^2 - p.ωq^2)]
    
    X  = [a1_ext + a1_ext', a2_ext + a2_ext']
    P  = [1im * (a1_ext' - a1_ext), 1im * (a2_ext' - a2_ext)]
    XP = ap_ext + ap_ext'
    
    sin_t  = sin(p.θ)
    cos_t  = cos(p.θ)
    sin_2t = sin(2*p.θ)
    
    H2       = 0.0 * Id_ext
    H3       = 0.0 * Id_ext
    H_filter = 0.0 * Id_ext

    for i in 1:2
        # --- Filter (1st Order) ---
        term_z_f = -2.0 * sin_t * (g[i] * gp[i] / ω[i]) * XP * σz_ext
        term_x_f = -cos_t * g[i] * gp[i] * A[i] * XP * σx_ext
        H_filter += term_z_f + term_x_f 
        
        # --- 2nd Order: Single index terms ---
        H2 += -2 * sin_t^2 * (g[i]^2 / ω[i]) * Id_ext          # [Sz, Vz]
        H2 += -cos_t^2 * g[i]^2 * A[i] * Id_ext                # [Sx, Vx] scalar part
        
        # --- 2nd Order: Double index terms ---
        for j in 1:2
            H2 += -cos_t^2 * g[i]*g[j] * B[i] * X[i]*X[j] * σz_ext                # [Sx, Vx] op part
            H2 += (sin_2t / 2) * (g[i]*g[j] / ω[i]) * commutator(P[i], X[j], anti = true) * σy_ext # [Sz, Vx]
            
            # [Sx, Vz]
            term_A = -(A[i] / 2) * commutator(P[i], X[j], anti = true) * σy_ext
            term_B = B[i] * X[i]*X[j] * σx_ext
            H2 += (sin_2t / 2) * g[i]*g[j] * (term_A + term_B)
        end
    end
    H2 *= 0.5 # Apply global 1/2 factor for 2nd order

    # --- 3rd Order: Double Sums ---
    for i in 1:2, k in 1:2
        # [Sz, [Sx, Vx]]
        H3 += 2 * sin_t * cos_t^2 * (g[i] * g[k]^2 / ω[k]) * (B[k] + B[i]) * X[i]
        # [Sz, [Sz, Vx]] (term 2)
        H3 += -sin_t * sin_2t * 2im * (g[i]^2 * g[k] / (ω[i]*ω[k])) * P[k] * σx_ext

        # [Sx, [Sz, Vx]] (term 2)
        term2_SxSzVx = 2* (g[i]^2 * g[k] / ω[i]) * (B[i] * X[k] + 1im * A[k] * P[k] * σz_ext)
        H3 += (cos_t * sin_2t / 2) * term2_SxSzVx
        
        # [Sx, [Sx, Vz]] (term 2)
        part_c = 2 * (2*B[i] + B[k]) * X[k]
        part_d = 2im * A[k] * P[k] * σz_ext
        term2_SxSxVz = A[i] * g[i]^2 * g[k] * (part_c + part_d)
        H3 -= (cos_t * sin_2t / 4) * term2_SxSxVz
    end

    # --- 3rd Order: Triple Sums ---
    for i in 1:2, j in 1:2, k in 1:2
        # [Sz, [Sz, Vx]] (term 1)
        H3 += -sin_t * sin_2t * (g[i]*g[j]*g[k] / (ω[i]*ω[k])) * commutator(P[k], P[i]*X[j], anti = true) * σx_ext
        
        # [Sz, [Sx, Vz]]
        term_A_1 = A[i] * commutator(P[k], P[i]*X[j], anti = true) * σx_ext
        term_B_1 = B[i] * commutator(P[k], X[i]*X[j], anti = true) * σy_ext
        term_C_1 = 2im * A[i] * (i == j ? 1.0 : 0.0) * P[k] * σx_ext
        H3 += sin_t * (sin_2t/2) * (g[i]*g[j]*g[k] / ω[k]) * (term_A_1 + term_B_1 + term_C_1)
        
        # [Sx, [Sx, Vx]]
        term_A_2 = A[k] * commutator(P[k], X[i]*X[j], anti = true) * σy_ext
        term_B_2 = -2 * B[k] * X[k]*X[i]*X[j] * σx_ext
        H3 += (cos_t^3 / 2) * g[i]*g[j]*g[k] * B[i] * (term_A_2 + term_B_2)
        
        # [Sx, [Sz, Vx]] (term 1)
        H3 += (cos_t * sin_2t / 2) * (g[i]*g[j]*g[k] / ω[i]) * A[k] * commutator(P[k], P[i]*X[j], anti = true) * σz_ext
        
        # [Sx, [Sx, Vz]] (term 1)
        part_a = A[i] * A[k] * commutator(P[k], P[i]*X[j], anti = true) * σz_ext
        part_b = 2 * B[i] * B[k] * X[k]*X[i]*X[j] * σz_ext
        H3 -= (cos_t * sin_2t / 4) * g[i]*g[j]*g[k] * (part_a + part_b)
    end
    H3 *= (1.0 / 3.0) 

    H_ext = H0 + Hint_P + H2 + H3 + H_filter

    P1_mat = sparse(I, N1, N1_ext)
    P2_mat = sparse(I, N2, N2_ext)
    Ip_mat = sparse(I, Np, Np)
    Iq_mat = sparse(I, Nq, Nq)
    
    P_full_mat = kron(P1_mat, kron(P2_mat, kron(Ip_mat, Iq_mat)))
    H_sub_mat = P_full_mat * H_ext.data * P_full_mat'
    
    H_sub = QuantumObject(H_sub_mat, type=Operator(), dims=dims_sys)
    
    return H_sub 
end

function H_num(p::SystemParams)
    H0 = p.ω1 * a1'*a1 + p.ω2 * a2'*a2 + p.ωp * ap'*ap + p.ωq * σz / 2
    Hint = (p.g1 * (a1+a1') + p.g2 * (a2+a2')) * (sin(p.θ) * σz + cos(p.θ) * σx)
    Hint_P = (p.g1p * (a1+a1') + p.g2p * (a2+a2')) * (ap+ap')

    S = SW_generator(p)
    H_1st_filter = commutator(S, Hint_P)
    H_2nd = 0.5 * commutator(S, Hint)
    H_3rd = (1.0/3.0) * commutator(S, commutator(S, Hint))  

    H_4th = (1.0/8.0) * commutator(S, commutator(S, commutator(S, Hint)))  
    H_2nd_filter = 0.5 * commutator(S, H_1st_filter)

    
    return H0 + Hint_P + H_1st_filter + H_2nd + H_3rd   + H_4th + H_2nd_filter
end

function SW_generator(p::SystemParams)
    A1 = 2*p.ω1/(p.ω1^2 - p.ωq^2)
    A2 = 2*p.ω2/(p.ω2^2 - p.ωq^2)
    B1 = 2*p.ωq/(p.ω1^2 - p.ωq^2)
    B2 = 2*p.ωq/(p.ω2^2 - p.ωq^2)
    Sz = sin(p.θ) * ( 
    (p.g1 / p.ω1) * (a1'-a1) * σz + 
    (p.g2 / p.ω2) * (a2'-a2) * σz 
    )
    Sx = (1.0 / 2.0) * cos(p.θ) * ( 
        p.g1 * (A1 * (a1'-a1) * σx - 1im* B1 * (a1'+a1) * σy) + 
        p.g2 * (A2 * (a2'-a2) * σx - 1im* B2 * (a2'+a2) * σy) 
    )
    S = Sz + Sx
    return S
end