using Test
using LatticeMC

@testset verbose = true "LatticeMC.jl" begin
    @testset verbose = true "Ising" begin
        include("./energy.jl")
        include("./ising.jl")
    end
    @testset verbose = true "AFQMC" begin
        include("./afqmc.jl")
        include("./afqmc_units.jl")
        include("./afqmc_convergence.jl")
    end
    @testset verbose = true "AFQMC ab initio" begin
        include("./ab_initio.jl")
    end
    @testset verbose = true "Heisenberg SSE" begin
        include("./heisenberg.jl")
    end
    @testset verbose = true "DQMC" begin
        include("./dqmc.jl")
    end
end