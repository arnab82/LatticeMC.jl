using LatticeMC
using Plots
using Random

include(joinpath(@__DIR__, "..", "test", "ed_reference.jl"))
include(joinpath(@__DIR__, "..", "test", "ab_initio_fci_reference.jl"))

Random.seed!(42)

# Equilibrium H4 (STO-3G) -- the case a single RHF determinant handles poorly
# (only ~10% of the correlation energy recovered; see afqmc_ab_initio_theory
# sections 6-7). A multi-determinant (CASCI) trial systematically closes the
# gap as determinants are added -- shown here directly.
mi = LatticeMC.build_h_chain_sto3g(1.4; n_atoms=4)
_, e_rhf = LatticeMC.rhf_scf(mi, 2)
e_fci = ab_initio_fci_ground_state_energy(mi.h1e, mi.h2e, 2, 2; E_nuc=mi.E_nuc)
println("E_RHF = $e_rhf")
println("E_FCI = $e_fci")

ks = [1, 2, 3, 5, 8, 12, 20]
energies = Float64[]
errors = Float64[]
captured = Float64[]
for k in ks
    trial, mi_mo, _ = LatticeMC.build_casci_trial(mi, 2, 2; max_dets=k)
    result = LatticeMC.run_afqmc_ab_initio(mi_mo, trial, 2, 2;
                                            dtau=0.01, num_walkers=200, num_steps=2500,
                                            equilibration_steps=500)
    push!(energies, result.energy_mean)
    push!(errors, result.energy_err)
    push!(captured, 100 * (e_rhf - result.energy_mean) / (e_rhf - e_fci))
    println("top-$k dets: E = $(round(result.energy_mean, digits=5)) +/- " *
            "$(round(result.energy_err, digits=5))   captured = $(round(captured[end], digits=1))%")
end

plt = plot(ks, energies, yerror=errors, marker=:circle, label="AFQMC (CASCI trial)",
           xlabel="number of trial determinants", ylabel="energy (Ha)",
           title="H4 (STO-3G, eq.): multi-det trial closes the phaseless gap",
           grid=true, dpi=800)
hline!(plt, [e_rhf], label="RHF", linestyle=:dot)
hline!(plt, [e_fci], label="FCI", linestyle=:dash)
savefig(plt, "afqmc_multidet_convergence.png")
display(plt)
