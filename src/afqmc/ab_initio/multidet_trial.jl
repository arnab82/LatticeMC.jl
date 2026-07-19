# Multi-determinant trial wavefunction |psi_T> = sum_i c_i |D_i>, each |D_i>
# a product of up/down Slater determinants (orbital matrices dets_up[i],
# dets_down[i]). A drop-in AbstractAbInitioTrial: everything the propagator
# and driver need (overlap, mixed Green's function, mixed local energy) is an
# overlap-weighted average over the determinants -- see the per-function
# derivations below and docs/src/afqmc_ab_initio_theory.md section 8.
#
# `ref_up`/`ref_down` is a single reference determinant used only to
# initialize the random walkers (typically the dominant CI determinant / the
# HF reference); it is *not* part of the trial expansion itself.
struct MultiDetTrial <: AbstractAbInitioTrial
    coeffs::Vector{ComplexF64}
    dets_up::Vector{Matrix{ComplexF64}}
    dets_down::Vector{Matrix{ComplexF64}}
    ref_up::Matrix{ComplexF64}
    ref_down::Matrix{ComplexF64}
end

function MultiDetTrial(coeffs, dets_up, dets_down)
    length(coeffs) == length(dets_up) == length(dets_down) ||
        throw(ArgumentError("coeffs, dets_up, dets_down must have equal length"))
    isempty(coeffs) && throw(ArgumentError("multi-determinant trial needs at least one determinant"))
    # walker-init reference = the largest-|coefficient| determinant
    iref = argmax(abs.(coeffs))
    return MultiDetTrial(collect(ComplexF64, coeffs), dets_up, dets_down,
                         copy(dets_up[iref]), copy(dets_down[iref]))
end

# <psi_T|phi> = sum_i c_i^* det(D_i^up' phi^up) det(D_i^down' phi^down).
function overlap_ab_initio(walker::AbInitioWalker, trial::MultiDetTrial)
    total = zero(ComplexF64)
    for i in eachindex(trial.coeffs)
        oup = overlap_ab_initio(walker.phi_up, trial.dets_up[i])
        odown = overlap_ab_initio(walker.phi_down, trial.dets_down[i])
        total += conj(trial.coeffs[i]) * oup * odown
    end
    return total
end

walker_overlap_ab_initio(walker::AbInitioWalker, trial::MultiDetTrial) = overlap_ab_initio(walker, trial)

# The single determinant used to seed walkers / size checks / the default
# energy shift; not part of the trial expansion (see the struct docstring).
reference_determinant(trial::MultiDetTrial) = (trial.ref_up, trial.ref_down)

# Multi-determinant trial: initialize every walker from the trial's single
# reference determinant (its dominant CI determinant); the full expansion
# enters only through the overlap/energy estimators, not the walker state.
function init_ab_initio_walkers(trial::MultiDetTrial, num_walkers::Int)
    return [AbInitioWalker(copy(trial.ref_up), copy(trial.ref_down), 1.0) for _ in 1:num_walkers]
end

# Mixed local energy for a multi-determinant trial:
#   E = sum_i c_i^* O_i E_i / sum_i c_i^* O_i,
# with O_i = <D_i|phi> the i-th determinant's overlap and E_i the *single-
# determinant* mixed local energy between D_i and the walker (its own
# Green's function). This is exact because <psi_T|H|phi> = sum_i c_i^*
# <D_i|H|phi> = sum_i c_i^* O_i E_i and <psi_T|phi> = sum_i c_i^* O_i, so
# the compact single-determinant machinery is reused per determinant and
# combined by overlap-weighting (theory doc section 8).
function local_energy_ab_initio(mi::MolecularIntegrals, walker::AbInitioWalker, trial::MultiDetTrial)
    numerator = zero(ComplexF64)
    denominator = zero(ComplexF64)
    for i in eachindex(trial.coeffs)
        oup = overlap_ab_initio(walker.phi_up, trial.dets_up[i])
        odown = overlap_ab_initio(walker.phi_down, trial.dets_down[i])
        O_i = conj(trial.coeffs[i]) * oup * odown
        (oup == 0 || odown == 0) && continue
        G_up = greens_function_ab_initio(walker.phi_up, trial.dets_up[i])
        G_down = greens_function_ab_initio(walker.phi_down, trial.dets_down[i])
        Gt = G_up .+ G_down
        e1 = tr(mi.h1e * Gt)
        e2 = zero(ComplexF64)
        Ns = mi.Ns
        for p in 1:Ns, q in 1:Ns, r in 1:Ns, s in 1:Ns
            v = mi.h2e[p, q, r, s]
            v == 0.0 && continue
            e2 += v * (Gt[q, p] * Gt[s, r] - G_up[s, p] * G_up[q, r] - G_down[s, p] * G_down[q, r])
        end
        e2 *= 0.5
        E_i = mi.E_nuc + e1 + e2
        numerator += O_i * E_i
        denominator += O_i
    end
    denominator == 0 && return ComplexF64(NaN)
    return numerator / denominator
end
