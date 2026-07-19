using LatticeMC
using Random
using Test

include("heisenberg_ed_reference.jl")

# --- SSE vs exact diagonalization on small bipartite lattices. At large
# beta the SSE energy converges to the ground state; compared to the exact
# Sz=0-sector ED for the same bond list. Tolerances accommodate the residual
# finite-beta bias + statistical error. ---
@testset "SSE vs ED on small bipartite lattices" begin
    cases = [
        ("4-site ring", LatticeMC.build_heisenberg_chain(4; pbc=true)),
        ("6-site ring", LatticeMC.build_heisenberg_chain(6; pbc=true)),
        ("2x2 plaquette", LatticeMC.build_heisenberg_square(2, 2)),
        ("2x4 cylinder", LatticeMC.build_heisenberg_square(2, 4)),
        ("2x3 open", LatticeMC.build_heisenberg_square(2, 3; pbc=false)),
    ]
    for (name, lat) in cases
        e_ed = heisenberg_ed_energy(lat.bonds, lat.Ns)
        r = LatticeMC.run_sse(lat; beta=32.0, num_thermal=3000, num_measure=40000, seed=7)
        @test isapprox(r.energy, e_ed; atol=0.02)
    end
end

# --- exact-energy deterministic checks (no ED build needed): the 4-site ring
# and the 2x2 plaquette are both 4-site rings with exact ground energy -2 J. ---
@testset "SSE recovers exact small-ring energies" begin
    for lat in (LatticeMC.build_heisenberg_chain(4; pbc=true),
                LatticeMC.build_heisenberg_square(2, 2))
        r = LatticeMC.run_sse(lat; beta=32.0, num_thermal=3000, num_measure=60000, seed=3)
        @test isapprox(r.energy, -2.0; atol=0.02)
    end
end

# --- the bipartite guard: run_sse must refuse a frustrated (non-bipartite)
# lattice rather than silently return a wrong answer. A 2x3 lattice with PBC
# has an odd (length-3) periodic direction -> non-bipartite. ---
@testset "run_sse rejects non-bipartite (frustrated) lattices" begin
    frustrated = LatticeMC.build_heisenberg_square(2, 3; pbc=true)   # odd PBC ring
    @test !LatticeMC.is_bipartite(frustrated)
    @test_throws ArgumentError LatticeMC.run_sse(frustrated; beta=8.0, num_measure=100)

    @test LatticeMC.is_bipartite(LatticeMC.build_heisenberg_square(4, 4; pbc=true))
    @test LatticeMC.is_bipartite(LatticeMC.build_heisenberg_chain(6; pbc=true))
end

# --- larger lattice against the literature: the 4x4 periodic square-lattice
# S=1/2 Heisenberg ground-state energy per site is -0.701780 (well-known
# exact value). Loose tolerance for finite-beta + statistical error. ---
@testset "SSE 4x4 vs literature ground-state energy per site" begin
    lat = LatticeMC.build_heisenberg_square(4, 4; pbc=true)
    r = LatticeMC.run_sse(lat; beta=24.0, num_thermal=3000, num_measure=40000, seed=11)
    @test isapprox(r.energy_per_site, -0.70178; atol=0.01)
end
