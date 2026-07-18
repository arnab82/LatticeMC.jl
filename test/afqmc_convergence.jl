using LatticeMC
using Random
using Test

include("ed_reference.jl")

# Real Monte Carlo noise is involved here, so assertions are kept loose
# (large effect sizes / generous tolerances) rather than tight bias-trend
# checks, to avoid flaky tests.

@testset "energy_err shrinks with more post-equilibration samples" begin
    Random.seed!(99)
    lattice = LatticeMC.build_hubbard_chain(4, 1.0, 4.0; pbc=false)
    trial = LatticeMC.build_trial_wavefunction(lattice, 2, 2)

    result_short = LatticeMC.run_afqmc(lattice, trial, 2, 2; dtau=0.01, num_walkers=100,
                                        num_steps=600, equilibration_steps=200)
    result_long = LatticeMC.run_afqmc(lattice, trial, 2, 2; dtau=0.01, num_walkers=100,
                                       num_steps=3000, equilibration_steps=200)

    # ~7x more post-equilibration samples in the long run; the expected
    # ~sqrt(7) shrink in the error bar comfortably dominates run-to-run noise
    # in the error *estimate* itself for a single seeded comparison.
    @test result_long.energy_err < result_short.energy_err
end

@testset "AFQMC vs ED agrees across different population-control frequencies" begin
    Random.seed!(123)
    lattice = LatticeMC.build_hubbard_chain(4, 1.0, 4.0; pbc=false)
    e_ed = ed_ground_state_energy(lattice, 2, 2)
    trial = LatticeMC.build_trial_wavefunction(lattice, 2, 2)

    for pc_every in (2, 50)
        result = LatticeMC.run_afqmc(lattice, trial, 2, 2; dtau=0.01, num_walkers=150,
                                      num_steps=2000, equilibration_steps=400,
                                      pop_control_every=pc_every)
        @test isapprox(result.energy_mean, e_ed; atol=0.6)
    end
end
