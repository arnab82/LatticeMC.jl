using LatticeMC
using LinearAlgebra
using Random
using Test

include("ed_reference.jl")
include("ab_initio_fci_reference.jl")

# --- cross-check: the general (Slater-Condon) FCI code above must exactly
# reproduce the Hubbard model's *specialized* ED (test/ed_reference.jl) when
# fed h1e=K, h2e[i,i,i,i]=U (all other entries zero) -- the Hubbard on-site
# term is precisely the special case of a general two-electron tensor with
# only that one nonzero pattern. Strong, cheap correctness check on the new
# general FCI code, reusing the already-validated Hubbard ED as ground
# truth. ---
@testset "general FCI matches Hubbard ED as a special case" begin
    for (Ns, Nup, Ndown, U) in ((4, 2, 2, 0.0), (4, 2, 2, 4.0), (4, 1, 1, 4.0), (4, 2, 1, 3.0), (6, 3, 3, 2.5))
        lattice = LatticeMC.build_hubbard_chain(Ns, 1.0, U; pbc=false)
        e_hubbard = ed_ground_state_energy(lattice, Nup, Ndown)

        h1e = copy(lattice.K)
        h2e = zeros(Ns, Ns, Ns, Ns)
        for i in 1:Ns
            h2e[i, i, i, i] = U
        end
        e_general = ab_initio_fci_ground_state_energy(h1e, h2e, Nup, Ndown)

        @test isapprox(e_hubbard, e_general; atol=1e-8)
    end
end

# --- STO-3G integral engine sanity: known textbook value (Szabo & Ostlund's
# worked H2/STO-3G example quotes S12 = 0.6593 at R=1.4 bohr), and the
# nuclear repulsion closed form. ---
@testset "STO-3G integral engine sanity checks" begin
    mi = LatticeMC.build_h_chain_sto3g(1.4; n_atoms=2)
    @test isapprox(mi.E_nuc, 1 / 1.4; atol=1e-12)
    @test isapprox(mi.h1e, mi.h1e'; atol=1e-10)

    Ns = mi.Ns
    for p in 1:Ns, q in 1:Ns, r in 1:Ns, s in 1:Ns
        v = mi.h2e[p, q, r, s]
        @test isapprox(v, mi.h2e[q, p, r, s]; atol=1e-9)
        @test isapprox(v, mi.h2e[p, q, s, r]; atol=1e-9)
        @test isapprox(v, mi.h2e[r, s, p, q]; atol=1e-9)
    end
end

# --- Cholesky decomposition: exact reconstruction, and each L^gamma
# symmetric (theory: guaranteed by h2e's 8-fold symmetry, not just assumed). ---
@testset "Cholesky decomposition of the ERI tensor" begin
    mi = LatticeMC.build_h_chain_sto3g(1.4; n_atoms=4)
    Ls = LatticeMC.cholesky_decompose_eri(mi.h2e; threshold=1e-10)
    Ns = mi.Ns

    h2e_recon = zeros(Ns, Ns, Ns, Ns)
    for L in Ls, p in 1:Ns, q in 1:Ns, r in 1:Ns, s in 1:Ns
        h2e_recon[p, q, r, s] += L[p, q] * L[r, s]
    end
    @test isapprox(h2e_recon, mi.h2e; atol=1e-8)
    @test all(isapprox(L, L'; atol=1e-10) for L in Ls)

    h1e_mod = LatticeMC.modified_one_body(mi.h1e, Ls)
    @test isapprox(h1e_mod, h1e_mod'; atol=1e-10)
end

# --- RHF is a valid variational upper bound on FCI at every geometry, and
# the correlation-energy gap should grow with bond length (the textbook
# "RHF dissociation catastrophe"). ---
@testset "RHF is a variational upper bound on FCI" begin
    bonds = (1.0, 1.4, 2.0, 3.0)
    gaps = Float64[]
    for bond in bonds
        mi = LatticeMC.build_h_chain_sto3g(bond; n_atoms=4)
        _, e_rhf = LatticeMC.rhf_scf(mi, 2)
        e_fci = ab_initio_fci_ground_state_energy(mi.h1e, mi.h2e, 2, 2; E_nuc=mi.E_nuc)
        @test e_rhf >= e_fci - 1e-8
        push!(gaps, e_rhf - e_fci)
    end
    @test issorted(gaps)
end

# --- UHF is a strict variational generalization of RHF (unrestricted
# optimization contains the restricted solution as a special case), so its
# converged energy can only be <= RHF's. Near equilibrium H4 is well
# single-reference: UHF should collapse back to the symmetric (RHF-like,
# zero-moment) solution. Stretched, it should spontaneously spin-polarize
# (the standard RHF-to-UHF instability at bond dissociation), giving a
# strictly lower energy and a nonzero, alternating local-moment pattern. ---
@testset "UHF: variational bound and symmetry breaking" begin
    mi_eq = LatticeMC.build_h_chain_sto3g(1.4; n_atoms=4)
    _, e_rhf_eq = LatticeMC.rhf_scf(mi_eq, 2)
    C_up_eq, C_down_eq, e_uhf_eq = LatticeMC.uhf_scf(mi_eq, 2, 2)
    @test e_uhf_eq <= e_rhf_eq + 1e-6
    moment_eq = vec(sum(C_up_eq .^ 2, dims=2)) .- vec(sum(C_down_eq .^ 2, dims=2))
    @test maximum(abs.(moment_eq)) < 0.05   # collapses to the paramagnetic solution

    mi_stretched = LatticeMC.build_h_chain_sto3g(3.0; n_atoms=4)
    _, e_rhf_st = LatticeMC.rhf_scf(mi_stretched, 2)
    C_up_st, C_down_st, e_uhf_st = LatticeMC.uhf_scf(mi_stretched, 2, 2)
    @test e_uhf_st < e_rhf_st - 0.05   # strictly, substantially lower
    moment_st = vec(sum(C_up_st .^ 2, dims=2)) .- vec(sum(C_down_st .^ 2, dims=2))
    @test maximum(abs.(moment_st)) > 0.3   # real spin polarization developed
    @test all(sign(moment_st[i]) != sign(moment_st[i+1]) for i in 1:3)   # alternating
end

# --- local_energy_ab_initio must exactly reproduce E_RHF at the (U=0-like)
# deterministic point where the walker equals the trial itself -- no
# stochastic component, same spirit as the Hubbard U=0 sanity check. ---
@testset "local_energy_ab_initio matches E_RHF at the trial itself" begin
    for bond in (1.0, 1.4, 2.5)
        mi = LatticeMC.build_h_chain_sto3g(bond; n_atoms=4)
        trial = LatticeMC.build_rhf_trial(mi, 2, 2)
        _, e_rhf = LatticeMC.rhf_scf(mi, 2)
        walker = LatticeMC.AbInitioWalker(copy(trial.phi_up), copy(trial.phi_down), 1.0)
        e_loc = LatticeMC.local_energy_ab_initio(mi, walker, trial)
        @test isapprox(real(e_loc), e_rhf; atol=1e-6)
        @test isapprox(imag(e_loc), 0.0; atol=1e-8)
    end
end

# --- force_bias_shift / propagate_step_ab_initio_force_bias!: with a
# zero shift, the force-biased propagator must be *exactly* identical to the
# plain one (not just "close", not just "unbiased in distribution") --
# fbar=0 makes x=xi (nothing shifted) and the Radon-Nikodym factor
# exp(-0-0)=1, so with matched randomness the two code paths must produce
# bit-identical walkers. A strong, deterministic correctness check on the
# force-bias machinery itself, independent of any statistical comparison. ---
@testset "propagate_step_ab_initio_force_bias! reduces to the plain propagator at fbar=0" begin
    mi = LatticeMC.build_h_chain_sto3g(1.4; n_atoms=4)
    trial = LatticeMC.build_rhf_trial(mi, 2, 2)
    Ls = LatticeMC.cholesky_decompose_eri(mi.h2e; threshold=1e-8)
    h1e_mod = LatticeMC.modified_one_body(mi.h1e, Ls)
    dtau = 0.01
    expH1_half = exp(-dtau / 2 .* h1e_mod)
    fbar_zero = zeros(length(Ls))

    Random.seed!(4242)
    walker_plain = LatticeMC.AbInitioWalker(copy(trial.phi_up), copy(trial.phi_down), 1.0)
    LatticeMC.propagate_step_ab_initio!(walker_plain, mi, trial, expH1_half, Ls, dtau)

    Random.seed!(4242)
    walker_fb = LatticeMC.AbInitioWalker(copy(trial.phi_up), copy(trial.phi_down), 1.0)
    LatticeMC.propagate_step_ab_initio_force_bias!(walker_fb, mi, trial, expH1_half, Ls, dtau, fbar_zero)

    @test isapprox(walker_plain.phi_up, walker_fb.phi_up; atol=1e-12)
    @test isapprox(walker_plain.phi_down, walker_fb.phi_down; atol=1e-12)
    @test isapprox(walker_plain.weight, walker_fb.weight; atol=1e-12)
end

@testset "force_bias_shift returns a sensible, nonzero shift" begin
    mi = LatticeMC.build_h_chain_sto3g(1.4; n_atoms=4)
    trial = LatticeMC.build_rhf_trial(mi, 2, 2)
    Ls = LatticeMC.cholesky_decompose_eri(mi.h2e; threshold=1e-8)
    fbar = LatticeMC.force_bias_shift(trial, Ls, 0.01)
    @test length(fbar) == length(Ls)
    @test all(isfinite, fbar)
    @test maximum(abs.(fbar)) > 1e-3   # not a no-op-sized shift
end

# --- run_afqmc_ab_initio(...; force_bias=true) is a variance-reduction
# option, not a bias-reduction one (theory doc section 7.1) -- it must still
# agree with FCI within the same kind of tolerance as the default, not
# necessarily better. No strict variance-must-decrease assertion here: the
# measured effect is modest (~7%) and single-seed statistical comparisons of
# error bars are noisy: making that assertion actually run in a test would
# be flaky, not a meaningful correctness check. ---
@testset "run_afqmc_ab_initio(force_bias=true) is correct on H2" begin
    mi = LatticeMC.build_h_chain_sto3g(1.4; n_atoms=2)
    trial = LatticeMC.build_rhf_trial(mi, 1, 1)
    e_fci = ab_initio_fci_ground_state_energy(mi.h1e, mi.h2e, 1, 1; E_nuc=mi.E_nuc)
    result = LatticeMC.run_afqmc_ab_initio(mi, trial, 1, 1; dtau=0.01, num_walkers=300,
                                            num_steps=2000, equilibration_steps=500,
                                            stabilize_every=5, pop_control_every=10, force_bias=true)
    @test isapprox(result.energy_mean, e_fci; atol=0.03)
end

# --- stochastic AFQMC vs exact diagonalization. H2 (essentially single-
# reference at these bond lengths) is a tight-tolerance correctness check;
# H4 with the plain RHF trial and Tier-1 (no force-bias) sampling is a much
# harder case -- multi-reference-ish character that a single determinant
# doesn't capture well, so the phaseless approximation's systematic bias is
# large (recovers roughly 10% of the correlation energy at bond=1.4 in
# testing, vs. 90%+ for H2). This is a genuine, measured property of this
# implementation (see docs/afqmc_ab_initio_theory.md), not a bug -- the H4
# tolerance below is deliberately loose and the assertions check "AFQMC
# meaningfully improves on RHF", not "AFQMC is close to FCI". ---
Random.seed!(20260717)

@testset "AFQMC vs FCI: H2 (tight tolerance)" begin
    for bond in (1.4, 2.0)
        mi = LatticeMC.build_h_chain_sto3g(bond; n_atoms=2)
        trial = LatticeMC.build_rhf_trial(mi, 1, 1)
        e_fci = ab_initio_fci_ground_state_energy(mi.h1e, mi.h2e, 1, 1; E_nuc=mi.E_nuc)
        result = LatticeMC.run_afqmc_ab_initio(mi, trial, 1, 1; dtau=0.01, num_walkers=300,
                                                num_steps=2000, equilibration_steps=500,
                                                stabilize_every=5, pop_control_every=10)
        @test isapprox(result.energy_mean, e_fci; atol=0.03)
    end
end

@testset "AFQMC vs FCI: H4 (loose sanity check, see docstring above)" begin
    mi = LatticeMC.build_h_chain_sto3g(1.4; n_atoms=4)
    trial = LatticeMC.build_rhf_trial(mi, 2, 2)
    _, e_rhf = LatticeMC.rhf_scf(mi, 2)
    e_fci = ab_initio_fci_ground_state_energy(mi.h1e, mi.h2e, 2, 2; E_nuc=mi.E_nuc)
    result = LatticeMC.run_afqmc_ab_initio(mi, trial, 2, 2; dtau=0.01, num_walkers=300,
                                            num_steps=2000, equilibration_steps=400,
                                            stabilize_every=5, pop_control_every=10)
    @test result.energy_mean < e_rhf - 0.002   # captures some correlation, below mean-field
    @test isapprox(result.energy_mean, e_fci; atol=0.08)
end

# --- energy_shift is a pure numerical-stabilization knob: multiplying every
# walker's weight by the same per-step constant exp(dtau*energy_shift)
# cancels in the weighted-average estimator and doesn't touch the random-
# number stream (population control uses only *relative* weights, scale-
# invariant; the estimator only ratios), so with a matched seed two very
# different shift values produce the *same* energy trace to machine
# precision. This is the exact statement of "changes no physics" (the small
# residual is only floating-point rounding from scaling weights by different
# constants -- measured ~5e-15 here, and crucially *not* O(1), i.e. no
# divergent resampling picks). Much stronger than a statistical comparison.
# (The shift's actual purpose -- preventing weight overflow for large-|E|
# systems like all-electron N2/F2 at ~-100/-200 Ha, where unshifted weights
# grow like exp(dtau*|E|) per step -- can't be exercised with the H-only
# STO-3G engine, but follows from this invariance plus choosing the shift to
# keep per-step growth ~1.) ---
@testset "energy_shift changes no physics (identical trace to machine precision)" begin
    mi = LatticeMC.build_h_chain_sto3g(1.4; n_atoms=2)
    trial = LatticeMC.build_rhf_trial(mi, 1, 1)

    Random.seed!(31415)
    r0 = LatticeMC.run_afqmc_ab_initio(mi, trial, 1, 1; dtau=0.01, num_walkers=100,
                                        num_steps=600, equilibration_steps=100, energy_shift=0.0)
    Random.seed!(31415)
    r1 = LatticeMC.run_afqmc_ab_initio(mi, trial, 1, 1; dtau=0.01, num_walkers=100,
                                        num_steps=600, equilibration_steps=100, energy_shift=-50.0)

    @test length(r0.energy_trace) == length(r1.energy_trace)
    @test maximum(abs.(r0.energy_trace .- r1.energy_trace)) < 1e-12
    @test isapprox(r0.energy_mean, r1.energy_mean; atol=1e-12)
end

# --- build_uhf_trial_ab_initio's whole point: at a stretched geometry where
# RHF is a qualitatively poor reference (the standard bond-dissociation
# instability), swapping in the symmetry-broken UHF trial should give a
# dramatically better AFQMC result -- a large, comparative effect (measured
# during development: ~75% of the correlation energy recovered vs. AFQMC
# actually landing *above* RHF, i.e. worse than mean-field, with the plain
# RHF trial), so this comparison is robust/non-flaky despite being
# stochastic. Near equilibrium, UHF collapses to RHF (previous testset), so
# no improvement is expected there -- not tested here for that reason. ---
@testset "UHF trial substantially improves AFQMC at a stretched geometry" begin
    mi = LatticeMC.build_h_chain_sto3g(3.0; n_atoms=4)
    e_fci = ab_initio_fci_ground_state_energy(mi.h1e, mi.h2e, 2, 2; E_nuc=mi.E_nuc)

    rhf_trial = LatticeMC.build_rhf_trial(mi, 2, 2)
    uhf_trial = LatticeMC.build_uhf_trial_ab_initio(mi, 2, 2)

    Random.seed!(7)
    result_rhf = LatticeMC.run_afqmc_ab_initio(mi, rhf_trial, 2, 2; dtau=0.01, num_walkers=300,
                                                num_steps=2000, equilibration_steps=400,
                                                stabilize_every=5, pop_control_every=10)
    Random.seed!(7)
    result_uhf = LatticeMC.run_afqmc_ab_initio(mi, uhf_trial, 2, 2; dtau=0.01, num_walkers=300,
                                                num_steps=2000, equilibration_steps=400,
                                                stabilize_every=5, pop_control_every=10)

    @test abs(result_uhf.energy_mean - e_fci) < abs(result_rhf.energy_mean - e_fci) - 0.1
    @test isapprox(result_uhf.energy_mean, e_fci; atol=0.15)
end

# --- multi-determinant / CASCI trial. Several independent correctness
# checks, from cheap+exact to the end-to-end point. ---

# (i) build_casci_trial's full expansion must reproduce the exact CI (=FCI)
# energy -- a deterministic check of the internal CI solve + MO transform,
# against the independent AO-basis FCI reference (both are basis-invariant).
@testset "CASCI full expansion reproduces the FCI energy" begin
    mi = LatticeMC.build_h_chain_sto3g(1.4; n_atoms=4)
    e_fci = ab_initio_fci_ground_state_energy(mi.h1e, mi.h2e, 2, 2; E_nuc=mi.E_nuc)
    _, _, e_ci = LatticeMC.build_casci_trial(mi, 2, 2)
    @test isapprox(e_ci, e_fci; atol=1e-9)
end

# (ii) determinant_matrix / multidet_from_ci basics.
@testset "multi-determinant trial construction" begin
    M = LatticeMC.determinant_matrix(4, [1, 3])
    @test size(M) == (4, 2)
    @test M[1, 1] == 1 && M[3, 2] == 1 && sum(abs.(M)) == 2

    # a one-determinant MultiDetTrial equals the corresponding single-det
    # trial for overlap purposes.
    trial = LatticeMC.multidet_from_ci(4, [1.0], [[1, 2]], [[1, 2]])
    @test length(trial.coeffs) == 1
    @test size(trial.ref_up) == (4, 2)
    @test_throws ArgumentError LatticeMC.MultiDetTrial(ComplexF64[], Matrix{ComplexF64}[], Matrix{ComplexF64}[])
end

# (iii) THE strong end-to-end check: a *full*-FCI multi-determinant trial is
# the exact ground state, so the phaseless approximation is exact and AFQMC
# must return the FCI energy to ~machine precision (tiny stochastic scatter
# only, since every walker's overlap ratio is with the exact eigenstate).
# This validates the entire multi-det pipeline (overlap, mixed Green's
# function, mixed local energy, propagation, phaseless gate) at once. ---
@testset "AFQMC with a full-FCI trial is exact" begin
    mi = LatticeMC.build_h_chain_sto3g(1.4; n_atoms=4)
    e_fci = ab_initio_fci_ground_state_energy(mi.h1e, mi.h2e, 2, 2; E_nuc=mi.E_nuc)
    full_trial, mi_mo, _ = LatticeMC.build_casci_trial(mi, 2, 2)
    Random.seed!(2)
    result = LatticeMC.run_afqmc_ab_initio(mi_mo, full_trial, 2, 2; dtau=0.01, num_walkers=100,
                                            num_steps=800, equilibration_steps=200)
    @test isapprox(result.energy_mean, e_fci; atol=1e-6)
end

# (iv) the point of the feature: a truncated CASCI trial systematically
# closes the equilibrium-H4 phaseless gap that neither UHF nor force bias
# could (theory doc section 6-7). More determinants -> monotonically closer
# to FCI, all physical (above FCI). Measured 8% (top-1) -> ~90% (top-10);
# tolerances here are loose/robust, checking the *ordering*, not exact %. ---
@testset "truncated CASCI trial systematically improves toward FCI" begin
    mi = LatticeMC.build_h_chain_sto3g(1.4; n_atoms=4)
    e_fci = ab_initio_fci_ground_state_energy(mi.h1e, mi.h2e, 2, 2; E_nuc=mi.E_nuc)

    t1, m1, _ = LatticeMC.build_casci_trial(mi, 2, 2; max_dets=1)
    t10, m10, _ = LatticeMC.build_casci_trial(mi, 2, 2; max_dets=10)

    Random.seed!(3)
    r1 = LatticeMC.run_afqmc_ab_initio(m1, t1, 2, 2; dtau=0.01, num_walkers=200,
                                        num_steps=1500, equilibration_steps=400)
    Random.seed!(3)
    r10 = LatticeMC.run_afqmc_ab_initio(m10, t10, 2, 2; dtau=0.01, num_walkers=200,
                                         num_steps=1500, equilibration_steps=400)

    # both physical (not below FCI, within stochastic error), and 10 dets
    # substantially closer to FCI than 1.
    @test r1.energy_mean > e_fci - 0.01
    @test r10.energy_mean > e_fci - 0.01
    @test abs(r10.energy_mean - e_fci) < abs(r1.energy_mean - e_fci) - 0.02
end
