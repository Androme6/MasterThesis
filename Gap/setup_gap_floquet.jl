using Random
using QuantumToolbox
using SciMLOperators: AddedOperator, MatrixOperator

QuantumToolbox._dense_similar(L::AddedOperator, args...) = QuantumToolbox._dense_similar(L.ops[1], args...)
QuantumToolbox._dense_similar(L::MatrixOperator, args...) = QuantumToolbox._dense_similar(L.A, args...)

struct GapPropagator{KWT}
    kwargs::KWT
end

struct GapArnoldiLindblad{KWT}
    kwargs::KWT
end


# 1. Define your keys as a Val type
const ODE_SOLVER_KWARGS = Val((:params, :abstol, :reltol, :maxiters, :alg))

# 2. Use a function that dispatches on the Val's type parameter
@generated function _extract_kwargs(nt::NamedTuple{KT}, ::Val{ODE_SOLVER_KWARGS}) where {KT, ODE_SOLVER_KWARGS}
    existing = Tuple(k for k in KT if k in ODE_SOLVER_KWARGS)
    
    return :(NamedTuple{$existing}(nt))
end


function _gap_eigensystem(gap_method::GapArnoldiLindblad, H, T, c_ops; kwargs...)
    # 1. Get the Hilbert space dimension N
    N = size(H.data, 1)
    
    # 2. Create a random N x N matrix on the correct backend
    ρ0_mat = QuantumToolbox._dense_similar(H.data, N, N)
    rand!(ρ0_mat)
    
    # 3. Create a valid standard Operator QuantumObject
    # H.dimensions perfectly describes an N x N Operator
    ρ0_op = QuantumObject(ρ0_mat, Operator(), H.dimensions)
    
    # 4. Use mat2vec to safely transform it into an OperatorKet
    # This automatically generates the correct 64-length vector and exact Liouville Dimensions
    ρ0 = mat2vec(ρ0_op)
    normalize!(ρ0.data)
    
    # 5. Solve
    eig_res = eigsolve_al(H, T, c_ops; ρ0 = ρ0, liouvillian_eigs = Val(false), gap_method.kwargs...)
    
    return Array(eig_res.values), eig_res.vectors, eig_res.converged, (eig_res.numops, eig_res.iter)
end

function _gap_eigensystem(gap_method::GapPropagator, H, T, c_ops; kwargs...)
    U_td = propagator(H, T, c_ops; _extract_kwargs(NamedTuple(kwargs), ODE_SOLVER_KWARGS)...)
    eig_res = eigenstates(U_td; gap_method.kwargs...)

    idxs = sortperm(Array(eig_res.values), by = abs2, rev = true)
    values = eig_res.values[idxs]
    vectors = eig_res.vectors[:, idxs]

    return Array(values), vectors, eig_res.converged, (eig_res.numops, eig_res.iter)
end

function gap_and_ss_population(H, c_ops, a, ωd; return_eigs_res::Val{RET_EIGS} = Val(false), e_ops = nothing, gap_method = nothing, kwargs...) where {RET_EIGS}
    T = abs(2π / ωd)
    T < 0 && throw(ArgumentError("Time T must be non-negative"))
    isinf(T) && throw(ArgumentError("Time T must be finite to compute the gap and steady state population."))

    # Time list for averaging over a period
    n_tlist = 100
    tlist = range(0, T, n_tlist)

    _gap_method = isnothing(gap_method) ? GapArnoldiLindblad(NamedTuple(kwargs)) : gap_method
    ϵ_td, ρ_td_vecs, eig_converged, eig_meta = _gap_eigensystem(_gap_method, H, T, c_ops; kwargs...)
    ρ_td = map(eachcol(ρ_td_vecs)) do data
        data_mat = vec2mat(data)

        diag_vals = Array(diag(data_mat))
        idx_max = findmax(abs, diag_vals)[2]
        θ_max = angle(diag_vals[idx_max])
        data_phase = data_mat * exp(-1im * θ_max)
        QuantumObject(data_phase, Operator(), H.dimensions)
    end
    idx1 = 1
    idx2 = 2 # findfirst(x -> abs(imag(x)) < 1e-6, ϵ_td[2:end]) # 2

    ρss_tmp = ρ_td[idx1] / tr(ρ_td[idx1])
    ρss = (ρss_tmp + ρss_tmp') / 2

    if !eig_converged
        @warn "Eigensolver did not converge"
        res = if RET_EIGS
            if !isnothing(e_ops)
                (NaN, NaN, eig_meta, [zeros(eltype(ρss), n_tlist) for _ in eachindex(e_ops)])
            else
                (NaN, NaN, eig_meta)
            end
        else
            if !isnothing(e_ops)
                (NaN, NaN, [zeros(eltype(ρss), n_tlist) for _ in eachindex(e_ops)])
            else
                (NaN, NaN)
            end
        end
        return res
    end

    δϵ = abs(ϵ_td[idx1] - ϵ_td[idx2])
    # if imag(ϵ_td[idx2]) > 1e-6
    #     _U_td = propagator(H, T, c_ops; params=[ωd])
    #     _ϵ_td, _ρ_td = eigenstates(_U_td)
    #     throw(ErrorException("Large imaginary part of the gap: $δϵ with ϵ_td = $ϵ_td, and _ϵ_td = $(reverse(_ϵ_td[end-3:end]))"))
    # end

    e_ops_vcat = isnothing(e_ops) ? [a' * a] : [a' * a, e_ops...]
    kwargs2 = _extract_kwargs(NamedTuple(kwargs), ODE_SOLVER_KWARGS)
    sol_me = mesolve(H, ρss, tlist, c_ops; e_ops = e_ops_vcat, progress_bar = Val(false), kwargs2...)
    averaged_expvals = dropdims(sum(sol_me.expect, dims=2), dims=2) ./ n_tlist

    if RET_EIGS
        if !isnothing(e_ops)
            return real(averaged_expvals[1]), δϵ, eig_meta, averaged_expvals[2:end]
        else
            return real(averaged_expvals[1]), δϵ, eig_meta
        end
    else
        if !isnothing(e_ops)
            return real(averaged_expvals[1]), δϵ, averaged_expvals[2:end]
        else
            return real(averaged_expvals[1]), δϵ
        end
    end
end

