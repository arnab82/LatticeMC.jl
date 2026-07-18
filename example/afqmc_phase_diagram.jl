using LatticeMC
using Plots
using Random

include(joinpath(@__DIR__, "..", "test", "ed_reference.jl"))

Random.seed!(42)

Ns = 4
t = 1.0
Nup, Ndown = 2, 2
Us = 0.0:1.0:8.0

afqmc_energies = Float64[]
afqmc_errors = Float64[]
ed_energies = Float64[]

for U in Us
    lattice = LatticeMC.build_hubbard_chain(Ns, t, U; pbc=false)
    trial = LatticeMC.build_trial_wavefunction(lattice, Nup, Ndown)
    result = LatticeMC.run_afqmc(lattice, trial, Nup, Ndown;
                                  dtau=0.01, num_walkers=200, num_steps=3000,
                                  equilibration_steps=500, stabilize_every=5,
                                  pop_control_every=10)
    push!(afqmc_energies, result.energy_mean)
    push!(afqmc_errors, result.energy_err)
    push!(ed_energies, ed_ground_state_energy(lattice, Nup, Ndown))

    println("U/t=$(U/t): AFQMC = $(result.energy_mean) +/- $(result.energy_err), ED = $(ed_energies[end])")
end

plt = plot(collect(Us) ./ t, ed_energies, label="Exact diagonalization", linewidth=2,
           xlabel="U/t", ylabel="Ground-state energy",
           title="Ns=$Ns chain (OBC), Nup=Ndown=$Nup", grid=true, dpi=800)
scatter!(plt, collect(Us) ./ t, afqmc_energies, yerror=afqmc_errors, label="AFQMC", markersize=4)
savefig(plt, "afqmc_phase_diagram.png")
display(plt)
