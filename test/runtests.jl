using Test
using LatticeMC

@testset "LatticeMC.jl" begin
    @testset "Ising" begin
        include("./energy.jl")
        include("./ising.jl")
    end
    @testset "AFQMC" begin
        include("./afqmc.jl")
        include("./afqmc_units.jl")
        include("./afqmc_convergence.jl")
    end
    @testset "AFQMC ab initio" begin
        include("./ab_initio.jl")
    end
end