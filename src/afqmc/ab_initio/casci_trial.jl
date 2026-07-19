# Build a multi-determinant trial from a (truncated) CI expansion.
#
# In the MO basis a determinant is just a choice of occupied orbitals, so its
# orbital matrix is the corresponding columns of the Ns x Ns identity --
# determinant_matrix below. `multidet_from_ci` assembles a MultiDetTrial from
# externally-supplied coefficients + occupation lists (e.g. a PySCF CASSCF CI
# vector), and `build_casci_trial` produces one from an internal full-CI
# solve for small systems (same Ns ceiling as exact diagonalization).

# occupied-orbital list (1-based, any order) -> Ns x length(occ) matrix whose
# columns are the corresponding identity columns.
function determinant_matrix(Ns::Int, occ::AbstractVector{<:Integer})
    M = zeros(ComplexF64, Ns, length(occ))
    for (col, orb) in enumerate(occ)
        M[orb, col] = 1.0
    end
    return M
end

function multidet_from_ci(Ns::Int, coeffs::AbstractVector, up_occs::AbstractVector, down_occs::AbstractVector)
    dets_up = [determinant_matrix(Ns, up_occs[i]) for i in eachindex(coeffs)]
    dets_down = [determinant_matrix(Ns, down_occs[i]) for i in eachindex(coeffs)]
    return MultiDetTrial(coeffs, dets_up, dets_down)
end

# --- compact internal FCI over the fixed-(Nup,Ndown) sector, MO basis ---
# Same bitmask second-quantized machinery as test/ab_initio_fci_reference.jl,
# duplicated here (small) because src/ cannot depend on test/. Bit i (0-based)
# of an Ns-bit integer = orbital i+1 occupied; fermion sign convention
# (-1)^(occupied orbitals below the target).
_occ_list(state::Int, Ns::Int) = [i + 1 for i in 0:Ns-1 if (state >> i) & 1 == 1]

function _casci_bit_states(Ns::Int, n::Int)
    return [s for s in 0:(2^Ns - 1) if count_ones(s) == n]
end

function _casci_sgn_annihilate(s::Int, i::Int)
    ((s >> i) & 1) == 0 && return (0, 0)
    sgn = isodd(count_ones(s & ((1 << i) - 1))) ? -1 : 1
    return (sgn, s & ~(1 << i))
end

function _casci_sgn_create(s::Int, i::Int)
    ((s >> i) & 1) == 1 && return (0, 0)
    sgn = isodd(count_ones(s & ((1 << i) - 1))) ? -1 : 1
    return (sgn, s | (1 << i))
end

function _casci_fci_ground_state(mi::MolecularIntegrals, Nup::Int, Ndown::Int)
    Ns = mi.Ns
    ups = _casci_bit_states(Ns, Nup)
    downs = _casci_bit_states(Ns, Ndown)
    ui = Dict(s => k for (k, s) in enumerate(ups))
    di = Dict(s => k for (k, s) in enumerate(downs))
    nu = length(ups); nd = length(downs)
    dim = nu * nd
    H = zeros(Float64, dim, dim)
    comb(iu, id) = (iu - 1) * nd + id

    for (iu, su) in enumerate(ups), p in 0:Ns-1, q in 0:Ns-1
        mi.h1e[p+1, q+1] == 0.0 && continue
        s1sgn, s1 = _casci_sgn_annihilate(su, q); s1sgn == 0 && continue
        s2sgn, s2 = _casci_sgn_create(s1, p); s2sgn == 0 && continue
        haskey(ui, s2) || continue
        c = mi.h1e[p+1, q+1] * s1sgn * s2sgn
        for id in 1:nd; H[comb(ui[s2], id), comb(iu, id)] += c; end
    end
    for (id, sd) in enumerate(downs), p in 0:Ns-1, q in 0:Ns-1
        mi.h1e[p+1, q+1] == 0.0 && continue
        s1sgn, s1 = _casci_sgn_annihilate(sd, q); s1sgn == 0 && continue
        s2sgn, s2 = _casci_sgn_create(s1, p); s2sgn == 0 && continue
        haskey(di, s2) || continue
        c = mi.h1e[p+1, q+1] * s1sgn * s2sgn
        for iu in 1:nu; H[comb(iu, di[s2]), comb(iu, id)] += c; end
    end
    for (iu, su) in enumerate(ups), p in 0:Ns-1, q in 0:Ns-1, r in 0:Ns-1, s in 0:Ns-1
        v = mi.h2e[p+1, q+1, r+1, s+1]; v == 0.0 && continue
        g1, s1 = _casci_sgn_annihilate(su, q); g1 == 0 && continue
        g2, s2 = _casci_sgn_annihilate(s1, s); g2 == 0 && continue
        g3, s3 = _casci_sgn_create(s2, r); g3 == 0 && continue
        g4, s4 = _casci_sgn_create(s3, p); g4 == 0 && continue
        haskey(ui, s4) || continue
        c = 0.5 * v * g1 * g2 * g3 * g4
        for id in 1:nd; H[comb(ui[s4], id), comb(iu, id)] += c; end
    end
    for (id, sd) in enumerate(downs), p in 0:Ns-1, q in 0:Ns-1, r in 0:Ns-1, s in 0:Ns-1
        v = mi.h2e[p+1, q+1, r+1, s+1]; v == 0.0 && continue
        g1, s1 = _casci_sgn_annihilate(sd, q); g1 == 0 && continue
        g2, s2 = _casci_sgn_annihilate(s1, s); g2 == 0 && continue
        g3, s3 = _casci_sgn_create(s2, r); g3 == 0 && continue
        g4, s4 = _casci_sgn_create(s3, p); g4 == 0 && continue
        haskey(di, s4) || continue
        c = 0.5 * v * g1 * g2 * g3 * g4
        for iu in 1:nu; H[comb(iu, di[s4]), comb(iu, id)] += c; end
    end
    for (iu, su) in enumerate(ups), p in 0:Ns-1, q in 0:Ns-1
        g1, s1 = _casci_sgn_annihilate(su, q); g1 == 0 && continue
        g2, s2 = _casci_sgn_create(s1, p); g2 == 0 && continue
        haskey(ui, s2) || continue
        for (id, sd) in enumerate(downs), r in 0:Ns-1, s in 0:Ns-1
            v = mi.h2e[p+1, q+1, r+1, s+1]; v == 0.0 && continue
            g3, t1 = _casci_sgn_annihilate(sd, s); g3 == 0 && continue
            g4, t2 = _casci_sgn_create(t1, r); g4 == 0 && continue
            haskey(di, t2) || continue
            H[comb(ui[s2], di[t2]), comb(iu, id)] += v * g1 * g2 * g3 * g4
        end
    end

    F = eigen(Symmetric(H))
    e0 = F.values[1]
    vec0 = F.vectors[:, 1]
    return e0 + mi.E_nuc, vec0, ups, downs, nd
end

# Transform integrals to a new orthonormal orbital basis given the full
# Ns x Ns coefficient matrix C (new_orbital = sum_mu C[mu, p] old_orbital_mu):
# h1e' = C' h1e C, and the standard 4-index transform of h2e (chemist
# notation preserved). E_nuc is basis-independent.
function mo_transform(mi::MolecularIntegrals, C::AbstractMatrix)
    Ns = mi.Ns
    h1e = C' * mi.h1e * C
    g = mi.h2e
    t1 = zeros(Ns, Ns, Ns, Ns)
    for s in 1:Ns, r in 1:Ns, q in 1:Ns, p in 1:Ns, a in 1:Ns
        t1[p, q, r, s] += C[a, p] * g[a, q, r, s]
    end
    t2 = zeros(Ns, Ns, Ns, Ns)
    for s in 1:Ns, r in 1:Ns, q in 1:Ns, p in 1:Ns, b in 1:Ns
        t2[p, q, r, s] += C[b, q] * t1[p, b, r, s]
    end
    t3 = zeros(Ns, Ns, Ns, Ns)
    for s in 1:Ns, r in 1:Ns, q in 1:Ns, p in 1:Ns, c in 1:Ns
        t3[p, q, r, s] += C[c, r] * t2[p, q, c, s]
    end
    h2e = zeros(Ns, Ns, Ns, Ns)
    for s in 1:Ns, r in 1:Ns, q in 1:Ns, p in 1:Ns, d in 1:Ns
        h2e[p, q, r, s] += C[d, s] * t3[p, q, r, d]
    end
    return MolecularIntegrals(Ns, h1e, h2e, mi.E_nuc)
end

"""
    build_casci_trial(mi, Nup, Ndown; max_dets=nothing, coeff_threshold=0.0)

Multi-determinant trial from a full-CI solve, built **in the RHF MO basis**
(this matters: a truncated CI expansion is only well-behaved in the MO basis,
where the HF determinant dominates -- truncating in a raw AO basis gives a
poor, unphysical trial). Runs RHF on `mi`, transforms the integrals to the MO
basis, solves full CI there, and keeps the largest-|coefficient|
determinants (truncated to `max_dets` and/or |coefficient| >=
`coeff_threshold`).

Returns `(trial, mi_mo, e_ci)`: the `MultiDetTrial`, the **MO-basis
integrals you must run AFQMC with** (trial and integrals have to share a
basis), and the exact CI energy. Closed-shell RHF reference, so `Nup ==
Ndown`. Small systems only (the CI solve has the exact-diagonalization size
ceiling); for larger systems supply an external expansion via
`multidet_from_ci` in whatever basis your walkers/integrals use.

    trial, mi_mo, e_ci = build_casci_trial(mi, n, n; max_dets=10)
    result = run_afqmc_ab_initio(mi_mo, trial, n, n)
"""
function build_casci_trial(mi::MolecularIntegrals, Nup::Int, Ndown::Int;
                            max_dets::Union{Int,Nothing}=nothing, coeff_threshold::Float64=0.0)
    Nup == Ndown || throw(ArgumentError("build_casci_trial requires a closed-shell RHF reference (Nup == Ndown)"))
    C, _ = rhf_scf(mi, Nup)
    mi_mo = mo_transform(mi, C)

    e_ci, vec0, ups, downs, nd = _casci_fci_ground_state(mi_mo, Nup, Ndown)
    Ns = mi_mo.Ns

    order = sortperm(abs.(vec0); rev=true)
    keep = coeff_threshold > 0 ? [n for n in order if abs(vec0[n]) >= coeff_threshold] : order
    max_dets !== nothing && (keep = keep[1:min(max_dets, length(keep))])

    coeffs = ComplexF64[]
    up_occs = Vector{Int}[]
    down_occs = Vector{Int}[]
    for n in keep
        iu = div(n - 1, nd) + 1
        id = mod(n - 1, nd) + 1
        push!(coeffs, vec0[n])
        push!(up_occs, _occ_list(ups[iu], Ns))
        push!(down_occs, _occ_list(downs[id], Ns))
    end

    return multidet_from_ci(Ns, coeffs, up_occs, down_occs), mi_mo, e_ci
end
