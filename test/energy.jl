using LatticeMC
using JLD2
using Test
function run()
    @load "lattice_n.jld2"
    @test LatticeMC.energy_manual(lattice_n) == enrg
end
run()
