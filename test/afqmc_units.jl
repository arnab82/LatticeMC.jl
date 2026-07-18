using LatticeMC
using LinearAlgebra
using Random
using Test

# --- Green's function: universal trace invariant (theory doc, section on
# the Green's function definition). tr(G) == N holds for *any* phi, psi_trial
# pair with nonzero overlap, not just at self-consistency, so it's a cheap,
# strong sanity check independent of any physical scenario. ---
@testset "tr(G) == N invariant" begin
    Random.seed!(1)
    lattice = LatticeMC.build_hubbard_chain(6, 1.0, 4.0; pbc=false)
    trial = LatticeMC.build_trial_wavefunction(lattice, 3, 2)
    phi_up = trial.phi_up .+ 0.1 .* randn(6, 3)
    phi_down = trial.phi_down .+ 0.1 .* randn(6, 2)

    G_up = LatticeMC.greens_function(phi_up, trial.phi_up)
    G_down = LatticeMC.greens_function(phi_down, trial.phi_down)

    @test isapprox(tr(G_up), 3.0; atol=1e-8)
    @test isapprox(tr(G_down), 2.0; atol=1e-8)
end

# --- the critical correctness test for the hand-derived Sherman-Morrison
# fast-update formulas: local_update_ratio + rank1_update_greens_function!
# must exactly reproduce what a brute-force recompute from scratch gives. ---
@testset "rank-1 update matches brute-force recompute" begin
    Random.seed!(42)
    lattice = LatticeMC.build_hubbard_chain(6, 1.0, 4.0; pbc=false)
    trial = LatticeMC.build_trial_wavefunction(lattice, 3, 3)
    phi = trial.phi_up .+ 0.05 .* randn(6, 3)
    G = LatticeMC.greens_function(phi, trial.phi_up)

    for i in 1:6
        gamma_factor = 0.7 + 0.6 * rand()
        R = LatticeMC.local_update_ratio(G, i, gamma_factor)

        phi_new = copy(phi)
        phi_new[i, :] .*= gamma_factor
        G_bruteforce = LatticeMC.greens_function(phi_new, trial.phi_up)

        G_fast = copy(G)
        LatticeMC.rank1_update_greens_function!(G_fast, i, gamma_factor, R)

        @test isapprox(G_fast, G_bruteforce; atol=1e-10)

        ov_before = LatticeMC.overlap(phi, trial.phi_up)
        ov_after = LatticeMC.overlap(phi_new, trial.phi_up)
        @test isapprox(R, ov_after / ov_before; atol=1e-10)
    end
end

# --- orthonormalize! changes the stored orbital matrix but must leave the
# *represented* Slater determinant's local energy exactly unchanged: G is
# invariant under phi -> phi*A for any invertible A (see afqmc_theory.md
# section 10), and thin-QR is exactly such a transform. ---
@testset "orthonormalize! preserves local_energy exactly" begin
    Random.seed!(7)
    lattice = LatticeMC.build_hubbard_chain(6, 1.0, 4.0; pbc=false)
    trial = LatticeMC.build_trial_wavefunction(lattice, 3, 2)
    walker = LatticeMC.Walker(trial.phi_up .+ 0.1 .* randn(6, 3),
                               trial.phi_down .+ 0.1 .* randn(6, 2), 1.0)

    e_before = LatticeMC.local_energy(lattice, walker, trial)
    LatticeMC.orthonormalize!(walker)
    e_after = LatticeMC.local_energy(lattice, walker, trial)

    @test isapprox(e_before, e_after; atol=1e-8)
end

# --- comb resampling must conserve total weight and population size exactly
# (theory doc, population control section). ---
@testset "population_control! conserves total weight and population size" begin
    Random.seed!(11)
    lattice = LatticeMC.build_hubbard_chain(4, 1.0, 2.0; pbc=false)
    trial = LatticeMC.build_trial_wavefunction(lattice, 2, 2)
    walkers = LatticeMC.init_walkers(trial, 20)
    for w in walkers
        w.weight = rand() * 3.0
    end
    total_before = sum(w.weight for w in walkers)
    n_before = length(walkers)

    LatticeMC.population_control!(walkers)

    @test length(walkers) == n_before
    @test isapprox(sum(w.weight for w in walkers), total_before; atol=1e-8)
end

# --- discrete Hirsch transform scalar identity (theory doc section 4):
# summing the two auxiliary-field branches must reproduce
# exp(-dtau*U*n_up*n_down) exactly, for each of the 4 single-site occupation
# states. ---
@testset "discrete Hirsch transform reproduces exp(-dtau*U*n_up*n_down)" begin
    U = 3.7
    dtau = 0.02
    lattice = LatticeMC.build_hubbard_chain(2, 1.0, U; pbc=false)
    gamma = LatticeMC.hs_gamma(lattice, dtau)

    for (nu, nd) in ((0, 0), (1, 0), (0, 1), (1, 1))
        lhs = 0.5 * sum(exp(-dtau * U / 2 * (nu + nd)) * exp(gamma * s * (nu - nd)) for s in (1, -1))
        rhs = exp(-dtau * U * nu * nd)
        @test isapprox(lhs, rhs; atol=1e-10)
    end
end

@testset "hs_gamma rejects attractive U" begin
    lattice = LatticeMC.build_hubbard_chain(4, 1.0, -1.0; pbc=false)
    @test_throws ArgumentError LatticeMC.hs_gamma(lattice, 0.01)
end

@testset "trial wavefunctions have orthonormal columns" begin
    lattice = LatticeMC.build_hubbard_chain(6, 1.0, 4.0; pbc=false)

    trial = LatticeMC.build_trial_wavefunction(lattice, 3, 2)
    @test isapprox(trial.phi_up' * trial.phi_up, I; atol=1e-10)
    @test isapprox(trial.phi_down' * trial.phi_down, I; atol=1e-10)

    uhf_trial = LatticeMC.build_uhf_trial(lattice, 3, 2)
    @test isapprox(uhf_trial.phi_up' * uhf_trial.phi_up, I; atol=1e-10)
    @test isapprox(uhf_trial.phi_down' * uhf_trial.phi_down, I; atol=1e-10)
end

@testset "bipartite_coloring" begin
    lattice_even = LatticeMC.build_hubbard_chain(6, 1.0, 4.0; pbc=true)
    coloring = LatticeMC.bipartite_coloring(lattice_even.K)
    @test coloring !== nothing
    @test all(abs.(coloring) .== 1)

    lattice_odd = LatticeMC.build_hubbard_chain(5, 1.0, 4.0; pbc=true)
    @test LatticeMC.bipartite_coloring(lattice_odd.K) === nothing
    @test_throws ArgumentError LatticeMC.build_uhf_trial(lattice_odd, 2, 2)
end

@testset "build_uhf_trial develops a staggered (AFM) moment at large U" begin
    lattice = LatticeMC.build_hubbard_chain(6, 1.0, 8.0; pbc=false)
    trial = LatticeMC.build_uhf_trial(lattice, 3, 3; m0=0.5)
    n_up = vec(sum(trial.phi_up .^ 2, dims=2))
    n_down = vec(sum(trial.phi_down .^ 2, dims=2))
    moment = n_up .- n_down

    @test maximum(abs.(moment)) > 0.1
    @test all(sign(moment[i]) != sign(moment[i+1]) for i in 1:5)
end
