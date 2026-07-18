using LatticeMC
using Random
using Statistics
using Test

# --- deterministic, zero-flakiness checks: closed-form energies for the
# fully-aligned and checkerboard configurations of the 2D square Ising
# model (periodic boundaries). Every site has 4 aligned (resp.
# anti-aligned) neighbors, so per-site energy is exactly -2J (resp. +2J). ---
@testset "closed-form energies: fully-aligned and checkerboard lattices" begin
    N = 6
    model = LatticeMC.IsingModel(N, 1.0, true)

    model.lattice = ones(Int, N, N)
    @test LatticeMC.energy_manual(model) == -2.0 * model.J * N^2
    @test LatticeMC.energy_periodic(model) == -2.0 * model.J * N^2

    model.lattice = [(-1)^(i + j) for i in 1:N, j in 1:N]
    @test LatticeMC.energy_manual(model) == 2.0 * model.J * N^2
    @test LatticeMC.energy_periodic(model) == 2.0 * model.J * N^2
end

# energy_manual (explicit branching) and energy_periodic (mod1) compute the
# same periodic-boundary sum, so they should agree exactly, not just
# approximately, on any lattice.
@testset "energy_manual and energy_periodic agree exactly" begin
    Random.seed!(5)
    for _ in 1:5
        N = rand(3:8)
        model = LatticeMC.IsingModel(N, 1.0, true)
        @test LatticeMC.energy_manual(model) == LatticeMC.energy_periodic(model)
    end
end

# Loose statistical check: deep in the ordered ferromagnetic phase
# (T << Tc = 2.269 J), the equilibrated magnetization magnitude should be
# large regardless of the (disordered) initial condition.
@testset "magnetization is large deep in the ordered phase" begin
    Random.seed!(2024)
    N = 10
    model = LatticeMC.IsingModel(N, 1.0, true)
    T = 1.0
    Bj = model.J / T
    total = 50_000
    skipped = 10_000

    spin_per_site, net_energy, net_spin = LatticeMC.metropolis(model, total, Bj, LatticeMC.energy_manual(model))
    m = abs(mean(spin_per_site[skipped:end]))
    @test m > 0.6
end
