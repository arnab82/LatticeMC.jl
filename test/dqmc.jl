using LatticeMC
using LinearAlgebra
using Random
using Test

include("dqmc_ed_reference.jl")

# --- stable Green's function: the UDT-stabilized (I + B_L...B_1)^{-1} must
# equal a naive dense computation where the naive one is still well-
# conditioned (small beta). A cheap, exact internal check. ---
@testset "DQMC stable Green's function matches naive at small beta" begin
    Random.seed!(1)
    lat = LatticeMC.build_hubbard_square(2, 2, 1.0, 4.0; pbc=false)
    Ns = lat.Ns
    dtau = 0.1
    lambda = LatticeMC.DQMC.dqmc_lambda(4.0, dtau)
    expK = exp(-dtau .* lat.K)
    L = 8
    field = rand((-1, 1), L, Ns)
    Bs = [LatticeMC.DQMC.dqmc_B(expK, lambda, view(field, l, :), 1) for l in 1:L]
    P = Matrix{Float64}(I, Ns, Ns)
    for l in 1:L
        P = Bs[l] * P
    end
    G_naive = inv(I + P)
    @test isapprox(LatticeMC.stable_greens(Bs), G_naive; atol=1e-10)
end

# --- DQMC energy vs the finite-temperature grand-canonical ED reference on a
# small half-filled Hubbard cluster, and the sign-free property (no negative-
# weight moves at half filling). Tolerance accommodates finite-dtau Trotter +
# statistical error; the Trotter part is checked to vanish separately below. ---
@testset "DQMC vs finite-T ED (half-filled 2x2 Hubbard)" begin
    lat = LatticeMC.build_hubbard_square(2, 2, 1.0, 4.0; pbc=false)
    for beta in (1.0, 2.0)
        e_ed = hubbard_thermal_energy(lat.K, lat.U, lat.Ns, beta)
        r = LatticeMC.run_dqmc(lat; beta=beta, L=round(Int, beta / 0.05),
                               num_thermal=400, num_sweeps=3000, seed=1)
        @test r.neg_move_fraction == 0.0          # sign-problem-free at half filling
        @test isapprox(r.energy, e_ed; atol=0.06)
    end
end

# --- Trotter convergence: the (systematic) dtau error must shrink as dtau
# decreases, confirming DQMC is exact in the dtau -> 0 limit (not merely
# "close"). ---
@testset "DQMC Trotter error shrinks with dtau" begin
    lat = LatticeMC.build_hubbard_square(2, 2, 1.0, 4.0; pbc=false)
    beta = 3.0
    e_ed = hubbard_thermal_energy(lat.K, lat.U, lat.Ns, beta)
    r_coarse = LatticeMC.run_dqmc(lat; beta=beta, L=30, num_thermal=300, num_sweeps=2000, seed=2)
    r_fine = LatticeMC.run_dqmc(lat; beta=beta, L=90, num_thermal=300, num_sweeps=2000, seed=3)
    @test abs(r_fine.energy - e_ed) < abs(r_coarse.energy - e_ed)
end
