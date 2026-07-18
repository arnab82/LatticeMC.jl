using LatticeMC
using LinearAlgebra
using Random
using Test

include("ed_reference.jl")

# --- deterministic sanity check: at U=0 the trial wavefunction IS the exact
# ground state, so the local energy of a single un-propagated walker must
# equal the sum of the Nup+Ndown lowest one-body eigenvalues, with no
# stochastic component at all. ---
@testset "AFQMC local energy matches free-fermion energy at U=0" begin
    lattice = LatticeMC.build_hubbard_chain(4, 1.0, 0.0; pbc=false)
    Nup, Ndown = 2, 2
    trial = LatticeMC.build_trial_wavefunction(lattice, Nup, Ndown)
    walker = LatticeMC.Walker(copy(trial.phi_up), copy(trial.phi_down), 1.0)

    e_local = LatticeMC.local_energy(lattice, walker, trial)
    e_exact = ed_ground_state_energy(lattice, Nup, Ndown)
    # atol accounts for the O(bias^2) energy shift from the tiny degeneracy-
    # lifting perturbation in build_trial_wavefunction (see its docstring).
    @test isapprox(e_local, e_exact; atol=1e-6)
end

# --- stochastic AFQMC vs exact diagonalization, small clusters, open
# boundary conditions (avoids the L=2-with-PBC doubled-hopping edge case). ---
Random.seed!(20240517)

function run_afqmc_case(lattice, Nup, Ndown)
    trial = LatticeMC.build_trial_wavefunction(lattice, Nup, Ndown)
    result = LatticeMC.run_afqmc(lattice, trial, Nup, Ndown;
                                  dtau=0.01, num_walkers=150, num_steps=2000,
                                  equilibration_steps=400, stabilize_every=5,
                                  pop_control_every=10)
    return result.energy_mean, result.energy_err
end

@testset "AFQMC vs exact diagonalization: 4-site chain (OBC)" begin
    for U in (0.0, 4.0)
        lattice = LatticeMC.build_hubbard_chain(4, 1.0, U; pbc=false)
        e_ed = ed_ground_state_energy(lattice, 2, 2)
        e_afqmc, e_err = run_afqmc_case(lattice, 2, 2)
        @test isapprox(e_afqmc, e_ed; atol=0.5)
    end
end

@testset "AFQMC vs exact diagonalization: 2x2 plaquette (OBC)" begin
    for U in (0.0, 4.0)
        lattice = LatticeMC.build_hubbard_square(2, 2, 1.0, U; pbc=false)
        e_ed = ed_ground_state_energy(lattice, 2, 2)
        e_afqmc, e_err = run_afqmc_case(lattice, 2, 2)
        @test isapprox(e_afqmc, e_ed; atol=0.5)
    end
end

@testset "AFQMC vs exact diagonalization: 6-site chain (OBC), half filling" begin
    for U in (0.0, 4.0)
        lattice = LatticeMC.build_hubbard_chain(6, 1.0, U; pbc=false)
        e_ed = ed_ground_state_energy(lattice, 3, 3)
        e_afqmc, e_err = run_afqmc_case(lattice, 3, 3)
        @test isapprox(e_afqmc, e_ed; atol=0.6)
    end
end

# PBC rings generically have doubly-degenerate single-particle levels away
# from the band edges (cos(k) == cos(-k)); safe to validate here because
# build_trial_wavefunction's bias robustly lifts that degeneracy (see its
# docstring) -- unlike the dimension-2-PBC doubled-hopping edge case (see
# lattice.jl), which only affects length-2 lattices, not length 6.
@testset "AFQMC vs exact diagonalization: 6-site chain (PBC), half filling" begin
    for U in (0.0, 4.0)
        lattice = LatticeMC.build_hubbard_chain(6, 1.0, U; pbc=true)
        e_ed = ed_ground_state_energy(lattice, 3, 3)
        e_afqmc, e_err = run_afqmc_case(lattice, 3, 3)
        @test isapprox(e_afqmc, e_ed; atol=0.6)
    end
end

@testset "AFQMC vs exact diagonalization: 4-site chain (OBC), quarter filling" begin
    lattice = LatticeMC.build_hubbard_chain(4, 1.0, 4.0; pbc=false)
    e_ed = ed_ground_state_energy(lattice, 1, 1)
    e_afqmc, e_err = run_afqmc_case(lattice, 1, 1)
    @test isapprox(e_afqmc, e_ed; atol=0.5)
end

@testset "AFQMC vs exact diagonalization: 4-site chain (OBC), asymmetric filling" begin
    lattice = LatticeMC.build_hubbard_chain(4, 1.0, 4.0; pbc=false)
    e_ed = ed_ground_state_energy(lattice, 2, 1)
    e_afqmc, e_err = run_afqmc_case(lattice, 2, 1)
    @test isapprox(e_afqmc, e_ed; atol=0.5)
end
