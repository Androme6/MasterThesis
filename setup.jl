using QuantumToolbox
using CairoMakie
using LaTeXStrings
using CUDA 
using SparseArrays
using Dates
using JLD2
using LinearAlgebra

const N1 = 20
const N2 = 4
const Np = 1
const Nq = 2
const dims_sys = (N1, N2, Np, Nq)

const a1 = QuantumToolbox.tensor(destroy(N1), qeye(N2), qeye(Np), qeye(Nq))
const a2 = QuantumToolbox.tensor(qeye(N1), destroy(N2), qeye(Np), qeye(Nq))
const ap = QuantumToolbox.tensor(qeye(N1), qeye(N2), destroy(Np), qeye(Nq))
const n1 = QuantumToolbox.tensor(num(N1), qeye(N2), qeye(Np), qeye(Nq))
const n2 = QuantumToolbox.tensor(qeye(N1), num(N2), qeye(Np), qeye(Nq))
const np = QuantumToolbox.tensor(qeye(N1), qeye(N2), num(Np), qeye(Nq))
const σz = QuantumToolbox.tensor(qeye(N1), qeye(N2), qeye(Np), sigmaz())
const σy = QuantumToolbox.tensor(qeye(N1), qeye(N2), qeye(Np), sigmay())
const σx = QuantumToolbox.tensor(qeye(N1), qeye(N2), qeye(Np), sigmax())
const Id = QuantumToolbox.tensor(qeye(N1), qeye(N2), qeye(Np), qeye(Nq))

const N1_ext = N1 + 2
const N2_ext = N2 + 2
    
const a1_ext = QuantumToolbox.tensor(destroy(N1_ext), qeye(N2_ext), qeye(Np), qeye(Nq))
const a2_ext = QuantumToolbox.tensor(qeye(N1_ext), destroy(N2_ext), qeye(Np), qeye(Nq))
const ap_ext = QuantumToolbox.tensor(qeye(N1_ext), qeye(N2_ext), destroy(Np), qeye(Nq))
const σz_ext = QuantumToolbox.tensor(qeye(N1_ext), qeye(N2_ext), qeye(Np), sigmaz())
const σy_ext = QuantumToolbox.tensor(qeye(N1_ext), qeye(N2_ext), qeye(Np), sigmay())
const σx_ext = QuantumToolbox.tensor(qeye(N1_ext), qeye(N2_ext), qeye(Np), sigmax())
const Id_ext = QuantumToolbox.tensor(qeye(N1_ext), qeye(N2_ext), qeye(Np), qeye(Nq))

const P1_mat = sparse(I, N1, N1_ext)
const P2_mat = sparse(I, N2, N2_ext)
const Ip_mat = sparse(I, Np, Np)
const Iq_mat = sparse(I, Nq, Nq)
const P_full_mat = kron(P1_mat, kron(P2_mat, kron(Ip_mat, Iq_mat)))


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
    ωd::Float64
end

function H_full(p::SystemParams)
    H0 = p.ω1 * a1'*a1 + p.ω2 * a2'*a2 + p.ωp * ap'*ap + p.ωq * σz / 2
    Hint = (p.g1 * (a1+a1') + p.g2 * (a2+a2')) * (sin(p.θ) * σz + cos(p.θ) * σx)
    Hint_P = (p.g1p * (a1+a1') + p.g2p * (a2+a2')) * (ap+ap')
    return H0 + Hint + Hint_P
end

function H_eff(p::SystemParams)
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
            H2 += (sin_2t / 2) * (g[i]*g[j] / ω[i]) * QuantumToolbox.commutator(P[i], X[j], anti = true) * σy_ext # [Sz, Vx]
            
            # [Sx, Vz]
            term_A = -(A[i] / 2) * QuantumToolbox.commutator(P[i], X[j], anti = true) * σy_ext
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
        H3 += -sin_t * sin_2t * (g[i]*g[j]*g[k] / (ω[i]*ω[k])) * QuantumToolbox.commutator(P[k], P[i]*X[j], anti = true) * σx_ext
        
        # [Sz, [Sx, Vz]]
        term_A_1 = A[i] * QuantumToolbox.commutator(P[k], P[i]*X[j], anti = true) * σx_ext
        term_B_1 = B[i] * QuantumToolbox.commutator(P[k], X[i]*X[j], anti = true) * σy_ext
        term_C_1 = 2im * A[i] * (i == j ? 1.0 : 0.0) * P[k] * σx_ext
        H3 += sin_t * (sin_2t/2) * (g[i]*g[j]*g[k] / ω[k]) * (term_A_1 + term_B_1 + term_C_1)
        
        # [Sx, [Sx, Vx]]
        term_A_2 = A[k] * QuantumToolbox.commutator(P[k], X[i]*X[j], anti = true) * σy_ext
        term_B_2 = -2 * B[k] * X[k]*X[i]*X[j] * σx_ext
        H3 += (cos_t^3 / 2) * g[i]*g[j]*g[k] * B[i] * (term_A_2 + term_B_2)
        
        # [Sx, [Sz, Vx]] (term 1)
        H3 += (cos_t * sin_2t / 2) * (g[i]*g[j]*g[k] / ω[i]) * A[k] * QuantumToolbox.commutator(P[k], P[i]*X[j], anti = true) * σz_ext
        
        # [Sx, [Sx, Vz]] (term 1)
        part_a = A[i] * A[k] * QuantumToolbox.commutator(P[k], P[i]*X[j], anti = true) * σz_ext
        part_b = 2 * B[i] * B[k] * X[k]*X[i]*X[j] * σz_ext
        H3 -= (cos_t * sin_2t / 4) * g[i]*g[j]*g[k] * (part_a + part_b)
    end
    H3 *= (1.0 / 3.0) 

    H_ext = H0 + Hint_P + H2 + H3 + H_filter
    H_sub_mat = P_full_mat * H_ext.data * P_full_mat'
    H_sub = QuantumObject(H_sub_mat, type=Operator(), dims=dims_sys)
    
    return H_sub 
end

# they match
function H_eff_4th_order(p::SystemParams)
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
    H4       = 0.0 * Id_ext
    H_filter = 0.0 * Id_ext

    for i in 1:2
        # --- Filter (1st Order) ---
        term_z_f = -2.0 * sin_t * (g[i] * gp[i] / ω[i]) * XP * σz_ext
        term_x_f = -cos_t * g[i] * gp[i] * A[i] * XP * σx_ext
        H_filter += term_z_f + term_x_f 
        
        # --- 2nd Order: Single index terms ---
        H2 += -2 * sin_t^2 * (g[i]^2 / ω[i]) * Id_ext
        H2 += -cos_t^2 * g[i]^2 * A[i] * Id_ext
        
        # --- 2nd Order: Double index terms ---
        for j in 1:2
            H2 += -cos_t^2 * g[i]*g[j] * B[i] * X[i]*X[j] * σz_ext
            H2 += (sin_2t / 2) * (g[i]*g[j] / ω[i]) * QuantumToolbox.commutator(P[i], X[j], anti = true) * σy_ext
            
            term_A = -(A[i] / 2) * QuantumToolbox.commutator(P[i], X[j], anti = true) * σy_ext
            term_B = B[i] * X[i]*X[j] * σx_ext
            H2 += (sin_2t / 2) * g[i]*g[j] * (term_A + term_B)
        end
    end
    H2 *= 0.5

    # --- 3rd Order: Double Sums ---
    for i in 1:2, k in 1:2
        H3 += 2 * sin_t * cos_t^2 * (g[i] * g[k]^2 / ω[k]) * (B[k] + B[i]) * X[i]
        H3 += -sin_t * sin_2t * 2im * (g[i]^2 * g[k] / (ω[i]*ω[k])) * P[k] * σx_ext

        term2_SxSzVx = 2* (g[i]^2 * g[k] / ω[i]) * (B[i] * X[k] + 1im * A[k] * P[k] * σz_ext)
        H3 += (cos_t * sin_2t / 2) * term2_SxSzVx
        
        part_c = 2 * (2*B[i] + B[k]) * X[k]
        part_d = 2im * A[k] * P[k] * σz_ext
        term2_SxSxVz = A[i] * g[i]^2 * g[k] * (part_c + part_d)
        H3 -= (cos_t * sin_2t / 4) * term2_SxSxVz
    end

    # --- 3rd Order: Triple Sums ---
    for i in 1:2, j in 1:2, k in 1:2
        H3 += -sin_t * sin_2t * (g[i]*g[j]*g[k] / (ω[i]*ω[k])) * QuantumToolbox.commutator(P[k], P[i]*X[j], anti = true) * σx_ext
        
        term_A_1 = A[i] * QuantumToolbox.commutator(P[k], P[i]*X[j], anti = true) * σx_ext
        term_B_1 = B[i] * QuantumToolbox.commutator(P[k], X[i]*X[j], anti = true) * σy_ext
        term_C_1 = 2im * A[i] * (i == j ? 1.0 : 0.0) * P[k] * σx_ext
        H3 += sin_t * (sin_2t/2) * (g[i]*g[j]*g[k] / ω[k]) * (term_A_1 + term_B_1 + term_C_1)
        
        term_A_2 = A[k] * QuantumToolbox.commutator(P[k], X[i]*X[j], anti = true) * σy_ext
        term_B_2 = -2 * B[k] * X[k]*X[i]*X[j] * σx_ext
        H3 += (cos_t^3 / 2) * g[i]*g[j]*g[k] * B[i] * (term_A_2 + term_B_2)
        
        H3 += (cos_t * sin_2t / 2) * (g[i]*g[j]*g[k] / ω[i]) * A[k] * QuantumToolbox.commutator(P[k], P[i]*X[j], anti = true) * σz_ext
        
        part_a = A[i] * A[k] * QuantumToolbox.commutator(P[k], P[i]*X[j], anti = true) * σz_ext
        part_b = 2 * B[i] * B[k] * X[k]*X[i]*X[j] * σz_ext
        H3 -= (cos_t * sin_2t / 4) * g[i]*g[j]*g[k] * (part_a + part_b)
    end
    H3 *= (1.0 / 3.0)

    # =========================================================
    # --- 4th Order: H4 = (1/8) [S,[S,[S, H_Rabi]]] ---
    # =========================================================

    # [Sz,[Sz,[Sx,Vx]]]: -4sin²cos² Σ_{ik} gi²gk²/(ωiωk)(Bi+Bk) σz
    for i in 1:2, k in 1:2
        H4 += -4 * sin_t^2 * cos_t^2 * (g[i]^2 * g[k]^2) / (ω[i] * ω[k]) * (B[i] + B[k]) * σz_ext
    end

    # [Sz,[Sz,[Sz,Vx]]]: -sin²sin2θ [ Σ_{ijkl} gigj gkgl/(ωlωiωk){Pl,{Pk,PiXj}}σy
    #                                 + Σ_{ikl}  4i glgi²gk/(ωkωiωl) PlPk σy ]
    for i in 1:2, j in 1:2, k in 1:2, l in 1:2
        inner = QuantumToolbox.commutator(P[k], P[i]*X[j], anti=true)
        outer = QuantumToolbox.commutator(P[l], inner, anti=true)
        H4 += -sin_t^2 * sin_2t * (g[i]*g[j]*g[k]*g[l]) / (ω[l]*ω[i]*ω[k]) * outer * σy_ext
    end
    for i in 1:2, k in 1:2, l in 1:2
        H4 += -sin_t^2 * sin_2t * (4im * g[l] * g[i]^2 * g[k]) / (ω[k]*ω[i]*ω[l]) * P[l]*P[k] * σy_ext
    end

    # [Sz,[Sz,[Sx,Vz]]]: sin²(sin2θ/2)[ Σ_{ijkl} gigj gkgl/(ωlωk)(Ai{Pl,{Pk,PiXj}}σy - Bi{Pl,{Pk,XiXj}}σx)
    #                                   + Σ_{ikl} 4igi²gkgl/(ωkωl) AiPkPl σy ]
    for i in 1:2, j in 1:2, k in 1:2, l in 1:2
        inner_PiXj = QuantumToolbox.commutator(P[k], P[i]*X[j], anti=true)
        outer_Pl_P = QuantumToolbox.commutator(P[l], inner_PiXj, anti=true)
        inner_XiXj = QuantumToolbox.commutator(P[k], X[i]*X[j], anti=true)
        outer_Pl_X = QuantumToolbox.commutator(P[l], inner_XiXj, anti=true)
        H4 += sin_t^2 * (sin_2t/2) * (g[i]*g[j]*g[k]*g[l]) / (ω[l]*ω[k]) * (
            A[i] * outer_Pl_P * σy_ext - B[i] * outer_Pl_X * σx_ext
        )
    end
    for i in 1:2, k in 1:2, l in 1:2
        H4 += sin_t^2 * (sin_2t/2) * (4im * g[i]^2 * g[k] * g[l]) / (ω[k]*ω[l]) * A[i] * P[k]*P[l] * σy_ext
    end

    # [Sz,[Sx,[Sx,Vx]]]: -(sinθcos³θ/2) Σ_{ijkl} gigj gkgl/ωl
    #                       [ AkBi{Pl,{Pk,XiXj}}σx + 2BkBi{Pl,XkXiXj}σy ]
    for i in 1:2, j in 1:2, k in 1:2, l in 1:2
        inner_XiXj  = QuantumToolbox.commutator(P[k], X[i]*X[j], anti=true)
        outer_Pl_XX = QuantumToolbox.commutator(P[l], inner_XiXj, anti=true)
        ac_Pl_XXX   = QuantumToolbox.commutator(P[l], X[k]*X[i]*X[j], anti=true)
        H4 += -(sin_t * cos_t^3 / 2) * (g[i]*g[j]*g[k]*g[l]) / ω[l] * (
            A[k]*B[i] * outer_Pl_XX * σx_ext + 2*B[k]*B[i] * ac_Pl_XXX * σy_ext
        )
    end

    # [Sz,[Sx,[Sz,Vx]]]: -4cos²sin²[ Σ_{ikl} gigkgl²/(ωiωl) AkPiPk
    #                                + Σ_{il}  gi²gl²/(ωiωl)  Bi σz ]
    for i in 1:2, k in 1:2, l in 1:2
        H4 += -4 * cos_t^2 * sin_t^2 * (g[i]*g[k]*g[l]^2) / (ω[i]*ω[l]) * A[k] * P[i]*P[k]
    end
    for i in 1:2, l in 1:2
        H4 += -4 * cos_t^2 * sin_t^2 * (g[i]^2 * g[l]^2) / (ω[i]*ω[l]) * B[i] * σz_ext
    end

    # [Sz,[Sx,[Sx,Vz]]]: 2cos²sin²[ Σ_{ikl} gigkgl²/ωl (AiAkPiPk + (BiBl+BkBl+BiBk)XiXk)
    #                               + Σ_{il}  gi²gl²/ωl  Ai(2Bi+Bl) σz ]
    for i in 1:2, k in 1:2, l in 1:2
        H4 += 2 * cos_t^2 * sin_t^2 * (g[i]*g[k]*g[l]^2) / ω[l] * (
            A[i]*A[k] * P[i]*P[k]
            + (B[i]*B[l] + B[k]*B[l] + B[i]*B[k]) * X[i]*X[k]
        )
    end
    for i in 1:2, l in 1:2
        H4 += 2 * cos_t^2 * sin_t^2 * (g[i]^2 * g[l]^2) / ω[l] * A[i] * (2*B[i] + B[l]) * σz_ext
    end

    # [Sx,[Sz,[Sx,Vx]]]: -2sinθcos³θ Σ_{ik} gi²gk²/ωk (Bi+Bk)Ai σx
    for i in 1:2, k in 1:2
        H4 += -2 * sin_t * cos_t^3 * (g[i]^2 * g[k]^2) / ω[k] * (B[i] + B[k]) * A[i] * σx_ext
    end

    # [Sx,[Sz,[Sz,Vx]]]: cos²sin²[ Σ_{ijkl} gigj gkgl/(ωiωk) Bl{Xl,{Pk,PiXj}}σz
    #                              + Σ_{ikl}  2gigkgl/(ωiωk) (igi Bl{Xl,Pk}σz + 2glAlPkPi) ]
    for i in 1:2, j in 1:2, k in 1:2, l in 1:2
        inner = QuantumToolbox.commutator(P[k], P[i]*X[j], anti=true)
        outer = QuantumToolbox.commutator(X[l], inner, anti=true)
        H4 += cos_t^2 * sin_t^2 * (g[i]*g[j]*g[k]*g[l]) / (ω[i]*ω[k]) * B[l] * outer * σz_ext
    end
    for i in 1:2, k in 1:2, l in 1:2
        ac_XlPk = QuantumToolbox.commutator(X[l], P[k], anti=true)
        H4 += cos_t^2 * sin_t^2 * 2 * (g[i]*g[k]*g[l]) / (ω[i]*ω[k]) * (
            1im * g[i] * B[l] * ac_XlPk * σz_ext
            + 2 * g[l] * A[l] * P[k]*P[i]
        )
    end

    # [Sx,[Sz,[Sx,Vz]]]: (cos²sin²/2)[ Σ_{ijkl} gigj gkgl/ωk (BiAl{Pl,{Pk,XiXj}} - AiBl{Xl,{Pk,PiXj}}) σz
    #                                  + Σ_{ikl}  2gigkgl (2BiBl gl/ωl XiXk
    #                                                      - 2AiAlgl/ωk PkPi
    #                                                      - igi AiBl/ωk {Xl,Pk} σz) ]
    for i in 1:2, j in 1:2, k in 1:2, l in 1:2
        inner_XX    = QuantumToolbox.commutator(P[k], X[i]*X[j], anti=true)
        outer_Pl_XX = QuantumToolbox.commutator(P[l], inner_XX, anti=true)
        inner_PX    = QuantumToolbox.commutator(P[k], P[i]*X[j], anti=true)
        outer_Xl_PX = QuantumToolbox.commutator(X[l], inner_PX, anti=true)
        H4 += (cos_t^2 * sin_t^2 / 2) * (g[i]*g[j]*g[k]*g[l]) / ω[k] * (
            B[i]*A[l] * outer_Pl_XX - A[i]*B[l] * outer_Xl_PX
        ) * σz_ext
    end
    for i in 1:2, k in 1:2, l in 1:2
        ac_XlPk = QuantumToolbox.commutator(X[l], P[k], anti=true)
        H4 += (cos_t^2 * sin_t^2 / 2) * 2 * g[i]*g[k]*g[l] * (
              (2*B[i]*B[l]*g[l] / ω[l]) * X[i]*X[k]
            - (2*A[i]*A[l]*g[l] / ω[k]) * P[k]*P[i]
            - (1im * g[i] * A[i] * B[l] / ω[k]) * ac_XlPk * σz_ext
        )
    end

    # [Sx,[Sx,[Sx,Vx]]]: (cos⁴/4)[ Σ_{ijkl} gigj gkgl Bi(AkAl{Pl,{Pk,XiXj}} + 4BkBl XiXjXkXl) σz
    #                              + Σ_{ijk}  4gigj gk² AkBi(3Bk+Bj) XiXj ]
    for i in 1:2, j in 1:2, k in 1:2, l in 1:2
        inner = QuantumToolbox.commutator(P[k], X[i]*X[j], anti=true)
        outer = QuantumToolbox.commutator(P[l], inner, anti=true)
        H4 += (cos_t^4 / 4) * g[i]*g[j]*g[k]*g[l] * B[i] * (
            A[k]*A[l] * outer + 4*B[k]*B[l] * X[i]*X[j]*X[k]*X[l]
        ) * σz_ext
    end
    for i in 1:2, j in 1:2, k in 1:2
        H4 += (cos_t^4 / 4) * 4 * g[i]*g[j]*g[k]^2 * A[k]*B[i] * (3*B[k] + B[j]) * X[i]*X[j]
    end

    # [Sx,[Sx,[Sz,Vx]]]: (cos³sin/2)[ Σ_{ijkl} gigj gkgl/ωi (AkBl{Xl,{Pk,PiXj}}σx - AkAl{Pl,{Pk,PiXj}}σy)
    #                                 + Σ_{ikl}  2igi²gkgl/ωi Ak(Bl{Xl,Pk}σx - 2AlPkPlσy)
    #                                 - Σ_{il}   4gi²gl²/ωi BiAl σx ]
    for i in 1:2, j in 1:2, k in 1:2, l in 1:2
        inner    = QuantumToolbox.commutator(P[k], P[i]*X[j], anti=true)
        outer_Xl = QuantumToolbox.commutator(X[l], inner, anti=true)
        outer_Pl = QuantumToolbox.commutator(P[l], inner, anti=true)
        H4 += (cos_t^3 * sin_t / 2) * (g[i]*g[j]*g[k]*g[l]) / ω[i] * (
            A[k]*B[l] * outer_Xl * σx_ext - A[k]*A[l] * outer_Pl * σy_ext
        )
    end
    for i in 1:2, k in 1:2, l in 1:2
        ac_XlPk = QuantumToolbox.commutator(X[l], P[k], anti=true)
        H4 += (cos_t^3 * sin_t / 2) * (2im * g[i]^2 * g[k] * g[l]) / ω[i] * A[k] * (
            B[l] * ac_XlPk * σx_ext - 2*A[l] * P[k]*P[l] * σy_ext
        )
    end
    for i in 1:2, l in 1:2
        H4 += -(cos_t^3 * sin_t / 2) * (4 * g[i]^2 * g[l]^2) / ω[i] * B[i] * A[l] * σx_ext
    end

    # [Sx,[Sx,[Sx,Vz]]]: (cos³sin/4)[ Σ_{ijkl} gigj gkgl (AiAkAl{Pl,{Pk,PiXj}} + 2AlBiBk{Pl,XkXiXj}) σy
    #                                 + Σ_{ikl}  4igi²gkgl AiAkAl PkPl σy
    #                                 - Σ_{ijkl} gigj gkgl Bl(AiAk{Xl,{Pk,PiXj}} + 4BiBk XiXjXkXl) σx
    #                                 - Σ_{ikl}  2igi²gkgl AiAkBl{Xl,Pk} σx
    #                                 + Σ_{il}   4gi²gl² AiAl(2Bi+Bl) σx ]
    for i in 1:2, j in 1:2, k in 1:2, l in 1:2
        inner_PX    = QuantumToolbox.commutator(P[k], P[i]*X[j], anti=true)
        outer_Pl_PX = QuantumToolbox.commutator(P[l], inner_PX, anti=true)
        ac_Pl_XXX   = QuantumToolbox.commutator(P[l], X[k]*X[i]*X[j], anti=true)
        outer_Xl_PX = QuantumToolbox.commutator(X[l], inner_PX, anti=true)
        # σy contributions
        H4 += (cos_t^3 * sin_t / 4) * g[i]*g[j]*g[k]*g[l] * (
            A[i]*A[k]*A[l] * outer_Pl_PX + 2*A[l]*B[i]*B[k] * ac_Pl_XXX
        ) * σy_ext
        # σx contributions
        H4 -= (cos_t^3 * sin_t / 4) * g[i]*g[j]*g[k]*g[l] * B[l] * (
            A[i]*A[k] * outer_Xl_PX + 4*B[i]*B[k] * X[i]*X[j]*X[k]*X[l]
        ) * σx_ext
    end
    for i in 1:2, k in 1:2, l in 1:2
        ac_XlPk = QuantumToolbox.commutator(X[l], P[k], anti=true)
        H4 += (cos_t^3 * sin_t / 4) * 4im  * g[i]^2 * g[k] * g[l] * A[i]*A[k]*A[l] * P[k]*P[l] * σy_ext
        H4 -= (cos_t^3 * sin_t / 4) * 2im  * g[i]^2 * g[k] * g[l] * A[i]*A[k]*B[l] * ac_XlPk * σx_ext
    end
    for i in 1:2, l in 1:2
        H4 += (cos_t^3 * sin_t / 4) * 4 * g[i]^2 * g[l]^2 * A[i]*A[l] * (2*B[i] + B[l]) * σx_ext
    end

    H4 *= (1.0 / 8.0)


    # =========================================================
    # --- 2nd Order Filter: H_filter2 = (1/2) [S,[S, H_filter]] ---
    # =========================================================
    H_filter2 = 0.0 * Id_ext
    
    for j in 1:2, k in 1:2
        # [Sz, [Sz, H_filter]] = 0
        
        # [Sz, [Sx, H_filter]]: -sin(2θ) * (gj*gk*gj,P / ωk) * Aj * Pk * XP * σy
        term_SzSx_f = -sin_2t * (g[j] * g[k] * gp[j] / ω[k]) * A[j] * P[k] * XP * σy_ext
        
        # [Sx, [Sz, H_filter]]: sin(2θ) * (gj*gk*gj,P / ωj) * XP * (Ak * Pk * σy - Bk * Xk * σx)
        term_SxSz_f = sin_2t * (g[j] * g[k] * gp[j] / ω[j]) * XP * (A[k] * P[k] * σy_ext - B[k] * X[k] * σx_ext)
        
        # [Sx, [Sx, H_filter]]: cos²(θ) * gj*gk*gj,P * Aj * Bk * Xk * XP * σz
        term_SxSx_f = cos_t^2 * g[j] * g[k] * gp[j] * A[j] * B[k] * X[k] * XP * σz_ext
        
        H_filter2 += term_SzSx_f + term_SxSz_f + term_SxSx_f
    end
    H_filter2 *= 0.5 # 1/2 from the BCH series expansion


    H_ext = H0 + Hint_P +H_filter + H_filter2 + H2 + H3 + H4 
    H_sub_mat = P_full_mat * H_ext.data * P_full_mat'
    H_sub = QuantumObject(H_sub_mat, type=Operator(), dims=dims_sys)
    
    return H_sub 
end
function H_num(p::SystemParams)
    H0 = p.ω1 * a1_ext'*a1_ext + p.ω2 * a2_ext'*a2_ext + p.ωp * ap_ext'*ap_ext + p.ωq * σz_ext / 2
    Hint_P = (p.g1p * (a1_ext+a1_ext') + p.g2p * (a2_ext+a2_ext')) * (ap_ext+ap_ext')
    Hint = (p.g1 * (a1_ext+a1_ext') + p.g2 * (a2_ext+a2_ext')) * (sin(p.θ) * σz_ext + cos(p.θ) * σx_ext)
    

    S = SW_generator(p)
    H_1st_filter = commutator(S, Hint_P)
    H_2nd = 0.5 * commutator(S, Hint)
    H_3rd = (1.0/3.0) * commutator(S, commutator(S, Hint))  

    H_4th = (1.0/8.0) * commutator(S, commutator(S, commutator(S, Hint)))  
    H_2nd_filter = 0.5 * commutator(S, H_1st_filter)

    H_ext = H0 + Hint_P + H_1st_filter + H_2nd + H_3rd + H_4th + H_2nd_filter
    H_sub_mat = P_full_mat * H_ext.data * P_full_mat'
    H_sub = QuantumObject(H_sub_mat, type=Operator(), dims=dims_sys)
    
    return H_sub
end


function SW_generator(p::SystemParams)
    A1 = 2*p.ω1/(p.ω1^2 - p.ωq^2)
    A2 = 2*p.ω2/(p.ω2^2 - p.ωq^2)
    B1 = 2*p.ωq/(p.ω1^2 - p.ωq^2)
    B2 = 2*p.ωq/(p.ω2^2 - p.ωq^2)
    Sz = sin(p.θ) * ( 
    (p.g1 / p.ω1) * (a1_ext'-a1_ext) * σz_ext + 
    (p.g2 / p.ω2) * (a2_ext'-a2_ext) * σz_ext 
    )
    Sx = (1.0 / 2.0) * cos(p.θ) * ( 
        p.g1 * (A1 * (a1_ext'-a1_ext) * σx_ext - 1im* B1 * (a1_ext'+a1_ext) * σy_ext) + 
        p.g2 * (A2 * (a2_ext'-a2_ext) * σx_ext - 1im* B2 * (a2_ext'+a2_ext) * σy_ext) 
    )
    S = Sz + Sx
    return S
end
# it matches with the commutator one
function L2_eff_4th_order(p::SystemParams, kp::Float64)
    # --- Setup Parameters ---
    g  = [p.g1, p.g2]
    ω  = [p.ω1, p.ω2]
    
    A  = [2*p.ω1/(p.ω1^2 - p.ωq^2), 2*p.ω2/(p.ω2^2 - p.ωq^2)]
    B  = [2*p.ωq/(p.ω1^2 - p.ωq^2), 2*p.ωq/(p.ω2^2 - p.ωq^2)]
    
    sin_t  = sin(p.θ)
    cos_t  = cos(p.θ)
    sin_2t = sin(2*p.θ)
    
    # --- Operators ---
    X = [a1_ext + a1_ext', a2_ext + a2_ext']
    P = [1im * (a1_ext' - a1_ext), 1im * (a2_ext' - a2_ext)]
    
    # Base Operator & Common Prefactor: L2 = -sqrt(k2/w2) * P2
    Γ = g[2] * B[2] * sqrt(kp / ω[2])
    L0 = -sqrt(kp / ω[2]) * P[2]
    
    # =========================================================
    # --- 1st Order Commutators ---
    # =========================================================
    f1 = -Γ * cos_t * σy_ext
    
    # =========================================================
    # --- 2nd Order Commutators ---
    # =========================================================
    f2 = 0.0 * Id_ext
    for k in 1:2
        f2 += Γ * sin_2t * (g[k] / ω[k]) * P[k] * σx_ext
        f2 -= Γ * cos_t^2 * g[k] * A[k] * P[k] * σz_ext
    end
    
    # =========================================================
    # --- 3rd Order Commutators ---
    # =========================================================
    f3 = 0.0 * Id_ext
    for j in 1:2, k in 1:2
        # [Sz, [Sz, [Sx, L2]]]
        f3 += 4 * Γ * sin_t^2 * cos_t * (g[j] * g[k] / (ω[j] * ω[k])) * P[j]*P[k] * σy_ext
        
        # [Sx, [Sz, [Sx, L2]]]
        ac_XP = QuantumToolbox.commutator(X[j], P[k], anti=true)
        f3 -= Γ * sin_t * cos_t^2 * (g[j] * g[k] / ω[k]) * B[j] * ac_XP * σz_ext
        
        # [Sx, [Sx, [Sx, L2]]]
        f3 += Γ * cos_t^3 * g[j] * g[k] * A[k] * A[j] * P[j]*P[k] * σy_ext
        f3 -= 0.5 * Γ * cos_t^3 * g[j] * g[k] * A[k] * B[j] * ac_XP * σx_ext
    end
    
    # =========================================================
    # --- 4th Order Commutators ---
    # =========================================================
    f4 = 0.0 * Id_ext
    
    # --- terms derived from C3(1): [Sz, [Sz, [Sx, L2]]] ---
    for j in 1:2, k in 1:2, l in 1:2
        f4 -= 8 * Γ * sin_t^3 * cos_t * (g[j]*g[k]*g[l] / (ω[j]*ω[k]*ω[l])) * P[j]*P[k]*P[l] * σx_ext
        f4 += 4 * Γ * sin_t^2 * cos_t^2 * (g[j]*g[k]*g[l] / (ω[j]*ω[k])) * A[l] * P[j]*P[k]*P[l] * σz_ext
    end
    # (FIX 1: Missing Identity Term)
    for j in 1:2, k in 1:2
        f4 += 8 * Γ * sin_t^2 * cos_t^2 * (g[j]^2 * g[k] / (ω[j]*ω[k])) * B[j] * P[k] * Id_ext
    end
    
    # --- terms derived from C3(2): [Sx, [Sz, [Sx, L2]]] ---
    # (FIX 2: Sign error on Identity term)
    for j in 1:2, k in 1:2
        f4 += 4 * Γ * sin_t^2 * cos_t^2 * (g[j]^2 * g[k] / (ω[j]*ω[k])) * B[j] * P[k] * Id_ext
    end
    for j in 1:2, k in 1:2, l in 1:2
        inner = QuantumToolbox.commutator(X[j], P[k], anti=true)
        ac_Pl = QuantumToolbox.commutator(P[l], inner, anti=true)
        ac_Xl = QuantumToolbox.commutator(X[l], inner, anti=true)
        
        f4 += 0.5 * Γ * sin_t * cos_t^3 * (g[j]*g[k]*g[l] / ω[k]) * B[j] * A[l] * ac_Pl * σy_ext
        f4 -= 0.5 * Γ * sin_t * cos_t^3 * (g[j]*g[k]*g[l] / ω[k]) * B[j] * B[l] * ac_Xl * σx_ext
    end
    
    # --- terms derived from C3(3): [Sx, [Sx, [Sx, L2]]] ---
    for j in 1:2, k in 1:2, l in 1:2
        inner = QuantumToolbox.commutator(X[j], P[k], anti=true)
        ac_Pl = QuantumToolbox.commutator(P[l], inner, anti=true)
        
        # (FIX 3: Sign error on σy term)
        f4 -= 0.5 * Γ * sin_t * cos_t^3 * (g[j]*g[k]*g[l] / ω[l]) * A[k] * B[j] * ac_Pl * σy_ext
        f4 -= 2.0 * Γ * sin_t * cos_t^3 * (g[j]*g[k]*g[l] / ω[l]) * A[k] * A[j] * P[j]*P[k]*P[l] * σx_ext
        
        ac_Xl = QuantumToolbox.commutator(X[l], inner, anti=true)
        f4 += Γ * cos_t^4 * g[j]*g[k]*g[l] * A[k] * A[j]*A[l] * P[j]*P[k]*P[l] * σz_ext
        f4 += 0.25 * Γ * cos_t^4 * g[j]*g[k]*g[l] * A[k] * B[j]*B[l] * ac_Xl * σz_ext
    end
    # (FIX 4: Coefficient correction on Identity term)
    for j in 1:2, k in 1:2
        f4 += 3 * Γ * cos_t^4 * g[j]^2 * g[k] * A[k] * A[j] * B[j] * P[k] * Id_ext
    end

    # =========================================================
    # --- Final Assembly ---
    # =========================================================
    # BCH Series: e^S L0 e^-S = L0 + f1 + 1/2 f2 + 1/6 f3 + 1/24 f4
    L_ext = L0 + f1 + (1.0/2.0)*f2 + (1.0/6.0)*f3 + (1.0/24.0)*f4
    
    L_sub_mat = P_full_mat * L_ext.data * P_full_mat'
    L_sub = QuantumObject(L_sub_mat, type=Operator(), dims=dims_sys)
    
    return L_sub 
end

# they don't match
function H_eff_RWA(p::SystemParams)
    # --- Setup Parameters ---
    g  = [p.g1, p.g2]
    gp = [p.g1p, p.g2p]
    ω  = [p.ω1, p.ω2]
    
    A  = [2*p.ω1/(p.ω1^2 - p.ωq^2), 2*p.ω2/(p.ω2^2 - p.ωq^2)]
    B  = [2*p.ωq/(p.ω1^2 - p.ωq^2), 2*p.ωq/(p.ω2^2 - p.ωq^2)]
    
    sin_t  = sin(p.θ)
    cos_t  = cos(p.θ)
    sin_2t = sin(2*p.θ)
    
    # --- Operators ---
    a = [a1_ext, a2_ext]
    n = [a1_ext'*a1_ext, a2_ext'*a2_ext]
    nP = ap_ext'*ap_ext
    
    # Bosonic RWA Helper Functions
    # 2n_i + 1
    n_2(i) = 2 * n[i] + Id_ext
    # 2n_i^2 + 2n_i + 1
    n_sq(i) = 2 * n[i]*n[i] + 2 * n[i] + Id_ext 

    # =========================================================
    # --- Free Hamiltonian & Filter (RWA) ---
    # =========================================================
    Δ1 = p.ω1 - p.ωd/2
    Δ2 = p.ω2 - p.ωd
    ΔP = p.ωp - p.ωd
    
    H0_RWA = Δ1 * n[1] + Δ2 * n[2] + ΔP * nP + (p.ωq / 2) * σz_ext
    H_filter_RWA = p.g2p * (a[2]' * ap_ext + a[2] * ap_ext')

    # =========================================================
    # --- 2nd Order Filter Commutators (RWA): (1/2)[S, [S, H_filter]] ---
    # =========================================================
    sum_filter = sum(g[j] * gp[j] * A[j] for j in 1:2)
    H_filter2_RWA = 0.5 * cos_t^2 * g[2] * B[2] * sum_filter * (a[2]' * ap_ext + a[2] * ap_ext') * σz_ext

    # =========================================================
    # --- 2nd Order Commutators (RWA): (1/2)[S, H_Rabi] ---
    # =========================================================
    H2_RWA = 0.0 * Id_ext
    for i in 1:2
        # [Sz, Vz]_RWA
        H2_RWA += 0.5 * (-2 * sin_t^2 * g[i]^2 / ω[i]) * Id_ext
        
        # [Sx, Vx]_RWA
        H2_RWA += 0.5 * (-cos_t^2 * g[i]^2 * A[i]) * Id_ext
        H2_RWA += 0.5 * (-cos_t^2 * g[i]^2 * B[i]) * n_2(i) * σz_ext
    end

    # =========================================================
    # --- 3rd Order Commutators (RWA): (1/3)[S, [S, H_Rabi]] ---
    # =========================================================
    # 3-wave mixing pumping mechanism: a1^2 a2' + (a1')^2 a2
    pump_op = a[1]*a[1]*a[2]' + a[1]'*a[1]'*a[2]
    
    term_3rd_Sz = cos_t * sin_2t * g[1]^2 * g[2] * ( (A[2] - A[1])/ω[1] + A[1]/ω[2] )
    term_3rd_Sx = -0.5 * cos_t * sin_2t * g[1]^2 * g[2] * ( A[1]*(2*A[2] - A[1]) + B[1]*(B[1] + 2*B[2]) )
    
    H3_RWA = (1.0 / 3.0) * (term_3rd_Sz + term_3rd_Sx) * pump_op * σz_ext

    # =========================================================
    # --- 4th Order Commutators (RWA): (1/8)[S, [S, [S, H_Rabi]]] ---
    # =========================================================
    H4_RWA = 0.0 * Id_ext

    # --- 4.1 From commutators with Sz outer generators ---
    # [Sz, [Sz, [Sx, Vx]]]
    for i in 1:2
        H4_RWA += -4 * sin_t^2 * cos_t^2 * (g[i]^4 / ω[i]^2) * 2 * B[i] * σz_ext
    end
    for i in 1:2, j in 1:2
        if i != j
            H4_RWA += -4 * sin_t^2 * cos_t^2 * (g[i]^2 * g[j]^2 / (ω[i]*ω[j])) * (B[i] + B[j]) * σz_ext
        end
    end
    
    # [Sz, [Sx, [Sz, Vx]]]
    for i in 1:2, j in 1:2
        H4_RWA += -4 * cos_t^2 * sin_t^2 * (g[i]^2 * g[j]^2 / (ω[i]*ω[j])) * A[i] * n_2(i)
        H4_RWA += -4 * cos_t^2 * sin_t^2 * (g[i]^2 * g[j]^2 / (ω[i]*ω[j])) * B[i] * σz_ext
    end

    # [Sz, [Sx, [Sx, Vz]]]
    for i in 1:2, j in 1:2
        H4_RWA += 2 * cos_t^2 * sin_t^2 * (g[i]^2 * g[j]^2 / ω[j]) * (A[i]^2 + 2*B[i]*B[j] + B[i]^2) * n_2(i)
        H4_RWA += 2 * cos_t^2 * sin_t^2 * (g[i]^2 * g[j]^2 / ω[j]) * A[i] * (2*B[i] + B[j]) * σz_ext
    end

    # --- 4.2 From commutators with Sx outer generators ---
    # [Sx, [Sz, [Sz, Vx]]]
    for i in 1:2
        H4_RWA += 4 * cos_t^2 * sin_t^2 * (g[i]^4 / ω[i]^2) * B[i] * n_sq(i) * σz_ext
    end
    for i in 1:2, j in 1:2
        if i != j
            H4_RWA += 4 * cos_t^2 * sin_t^2 * (g[i]^2 * g[j]^2 / ω[i]^2) * B[j] * n_2(i) * n_2(j) * σz_ext
        end
        H4_RWA += 4 * cos_t^2 * sin_t^2 * (g[i]^2 * g[j]^2 / ω[i]^2) * A[j] * n_2(i)
    end

    # [Sx, [Sz, [Sx, Vz]]]
    for i in 1:2, j in 1:2
        if i != j
            H4_RWA += 2 * cos_t^2 * sin_t^2 * g[i]^2 * g[j]^2 * (B[i]*A[j]/ω[j] - A[i]*B[j]/ω[i]) * n_2(i) * n_2(j) * σz_ext
        end
        H4_RWA += 2 * cos_t^2 * sin_t^2 * (g[i]^2 * g[j]^2 / ω[j]) * (2*B[i]*B[j] - A[i]*A[j]) * n_2(i)
    end

    # [Sx, [Sx, [Sx, Vx]]]
    for i in 1:2
        H4_RWA += cos_t^4 * g[i]^4 * B[i] * (A[i]^2 + 3*B[i]^2) * n_sq(i) * σz_ext
    end
    for i in 1:2, j in 1:2
        if i != j
            H4_RWA += cos_t^4 * g[i]^2 * g[j]^2 * (B[i]*A[j]^2 + B[i]*B[j]^2 + 2*B[i]^2*B[j]) * n_2(i) * n_2(j) * σz_ext
        end
        H4_RWA += cos_t^4 * g[i]^2 * g[j]^2 * A[j]*B[i] * (3*B[j] + B[i]) * n_2(i)
    end

    H4_RWA *= (1.0 / 8.0)

    # =========================================================
    # --- Final Assembly ---
    # =========================================================
    H_ext_RWA = H0_RWA + H_filter_RWA + H_filter2_RWA + H2_RWA + H3_RWA + H4_RWA
    
    H_sub_mat = P_full_mat * H_ext_RWA.data * P_full_mat'
    H_sub = QuantumObject(H_sub_mat, type=Operator(), dims=dims_sys)
    
    return H_sub 
end
function H_eff_num_RWA(p::SystemParams)
    # 1. Get the full 4th-order effective model (in the LAB frame)
    H_eff_qobj = H_eff_4th_order(p)
    
    # 2. Build the RWA conservation operators
    N_ex_ext = (a1_ext'*a1_ext) + 2*(a2_ext'*a2_ext) + 2*(ap_ext'*ap_ext)
    N_q_ext  = σz_ext
    
    N_ex_sub = P_full_mat * N_ex_ext.data * P_full_mat'
    N_q_sub  = P_full_mat * N_q_ext.data * P_full_mat'
    
    N_ex_diag = round.(real.(diag(N_ex_sub)), digits=3)
    N_q_diag  = round.(real.(diag(N_q_sub)), digits=3)
    
    # 3. Scrub the Hamiltonian matrix
    H_dense = Array(H_eff_qobj.data)
    dim = size(H_dense, 1)
    
    for i in 1:dim
        for j in 1:dim
            if (N_ex_diag[i] != N_ex_diag[j]) || (N_q_diag[i] != N_q_diag[j])
                H_dense[i, j] = 0.0
            end
        end
    end
    
    # =========================================================
    # 4. FIX: Shift to the Rotating Frame!
    # Mode 1 rotates at ωd/2, Mode 2 at ωd, Mode P at ωd
    # =========================================================
    H_rot_ext = (p.ωd / 2) * (a1_ext'*a1_ext) + p.ωd * (a2_ext'*a2_ext) + p.ωd * (ap_ext'*ap_ext)
    H_rot_sub = P_full_mat * H_rot_ext.data * P_full_mat'
    
    # Subtract the rotation to get the exact RWA H0 (Δ1, Δ2, ΔP)
    H_dense_rot = H_dense - Array(H_rot_sub)

    return QuantumObject(sparse(H_dense_rot), type=Operator(), dims=dims_sys)
end

# they match
function L2_eff_RWA(p::SystemParams, kp::Float64)
    # --- Setup Parameters ---
    g  = [p.g1, p.g2]
    ω  = [p.ω1, p.ω2]
    
    A  = [2*p.ω1/(p.ω1^2 - p.ωq^2), 2*p.ω2/(p.ω2^2 - p.ωq^2)]
    B  = [2*p.ωq/(p.ω1^2 - p.ωq^2), 2*p.ωq/(p.ω2^2 - p.ωq^2)]
    
    sin_t = sin(p.θ)
    cos_t = cos(p.θ)
    
    # --- Operators ---
    a = [a1_ext, a2_ext]
    n = [a1_ext'*a1_ext, a2_ext'*a2_ext]
    
    # Base Common Prefactor
    Γ = g[2] * B[2] * sqrt(kp / ω[2])
    
    C_jumps = []
    
    for m in 1:2
        # =========================================================
        # --- 0th & 2nd Order ---
        # =========================================================
        # L0 = -sqrt(kp/w2) P2. The pure annihilation path is (+1im * a).
        C0 = (m == 2) ? 1im * sqrt(kp / ω[2]) * a[2] : 0.0 * Id_ext
        
        # f2 = - Γ * cos_t^2 * gm * Am * P_m * σz 
        # Map P_m -> (+1im * a_m) and apply BCH factor (1/2)
        C2 = 0.5 * 1im * Γ * cos_t^2 * g[m] * A[m] * σz_ext * a[m]
        
        # =========================================================
        # --- 4th Order ---
        # =========================================================
        term_I = 0.0 * Id_ext
        term_Z = 0.0 * Id_ext
        
        for j in 1:2
            # --- Scalar (Identity) Shifts ---
            term_I += 12 * sin_t^2 * (g[j]^2 * g[m] / (ω[j] * ω[m])) * B[j] * Id_ext
            term_I += 3 * cos_t^2 * (g[j]^2 * g[m]) * A[m] * A[j] * B[j] * Id_ext
            
            # --- Photon-Number Dependent Shifts (σz) ---
            if j != m
                # Cross-scattering: 3 permutations
                n_cross = 2 * n[j] + Id_ext
                
                term_Z += 4 * sin_t^2 * (g[j]^2 * g[m]) * (A[m]/(ω[j]^2) + 2*A[j]/(ω[j]*ω[m])) * n_cross
                term_Z += cos_t^2 * (g[j]^2 * g[m]) * A[m] * (3 * A[j]^2 + B[j]^2) * n_cross
            else
                # Self-scattering: 1 permutation
                n_self = n[m] + Id_ext
                
                term_Z += 12 * sin_t^2 * (g[m]^3) * (A[m]/(ω[m]^2)) * n_self
                term_Z += cos_t^2 * (g[m]^3) * A[m] * (3 * A[m]^2 + B[m]^2) * n_self
            end
        end
        
        # All P contractions in 4th order map to (-1im * a_m). Apply BCH factor (1/24)
        C4 = (1.0 / 24.0) * (-1im) * Γ * cos_t^2 * (term_I + term_Z * σz_ext) * a[m]
        
        # =========================================================
        # --- Final Assembly ---
        # =========================================================
        Cm_ext = C0 + C2 + C4
        
        Cm_sub_mat = P_full_mat * Cm_ext.data * P_full_mat'
        Cm_sub = QuantumObject(Cm_sub_mat, type=Operator(), dims=dims_sys)
        push!(C_jumps, Cm_sub)
    end
    
    return C_jumps[1], C_jumps[2]
end
function L2_eff_num_RWA(L_eff_qobj::QuantumObject)
    L_dense = Array(L_eff_qobj.data)
    dim = size(L_dense, 1)
    
    # 1. Get the diagonals of the number operators in the target subspace
    n1_diag = round.(real.(diag(P_full_mat * (a1_ext'*a1_ext).data * P_full_mat')), digits=3)
    n2_diag = round.(real.(diag(P_full_mat * (a2_ext'*a2_ext).data * P_full_mat')), digits=3)
    np_diag = round.(real.(diag(P_full_mat * (ap_ext'*ap_ext).data * P_full_mat')), digits=3)
    nq_diag = round.(real.(diag(P_full_mat * σz_ext.data * P_full_mat')), digits=3)

    C1_dense = zeros(ComplexF64, dim, dim)
    C2_dense = zeros(ComplexF64, dim, dim)

    # 2. Extract the pure annihilation paths
    for i in 1:dim
        for j in 1:dim
            # L_dense[i, j] represents a transition from state |j> to state |i>
            
            # Mode 1 Jump: Final state |i> has 1 less photon in Mode 1 than initial state |j>
            if (n1_diag[i] == n1_diag[j] - 1) && (n2_diag[i] == n2_diag[j]) && 
               (np_diag[i] == np_diag[j]) && (nq_diag[i] == nq_diag[j])
                C1_dense[i, j] = L_dense[i, j]
            end

            # Mode 2 Jump: Final state |i> has 1 less photon in Mode 2 than initial state |j>
            if (n2_diag[i] == n2_diag[j] - 1) && (n1_diag[i] == n1_diag[j]) && 
               (np_diag[i] == np_diag[j]) && (nq_diag[i] == nq_diag[j])
                C2_dense[i, j] = L_dense[i, j]
            end
        end
    end

    return QuantumObject(sparse(C1_dense), type=Operator(), dims=dims_sys), 
           QuantumObject(sparse(C2_dense), type=Operator(), dims=dims_sys)
end


function H_drive_num_RWA(H_drive_qobj::QuantumObject)
    # Get diagonals of the effective excitation number and qubit state
    N_ex_diag = round.(real.(diag(P_full_mat * ((a1_ext'*a1_ext) + 2*(a2_ext'*a2_ext) + 2*(ap_ext'*ap_ext)).data * P_full_mat')), digits=3)
    N_q_diag  = round.(real.(diag(P_full_mat * σz_ext.data * P_full_mat')), digits=3)

    H_d_dense = Array(H_drive_qobj.data)
    dim = size(H_d_dense, 1)

    for i in 1:dim
        for j in 1:dim
            # Keep ONLY terms that create/destroy exactly 2 effective quanta (the drive frequency)
            # and leave the qubit in its current state.
            ΔN_ex = abs(N_ex_diag[i] - N_ex_diag[j])
            ΔN_q  = abs(N_q_diag[i] - N_q_diag[j])
            
            if (ΔN_ex != 2) || (ΔN_q != 0)
                H_d_dense[i, j] = 0.0
            end
        end
    end

    return QuantumObject(sparse(H_d_dense), type=Operator(), dims=dims_sys)
end