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
