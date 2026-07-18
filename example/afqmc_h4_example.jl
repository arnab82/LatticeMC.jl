using LatticeMC
using Plots
using Random

include(joinpath(@__DIR__, "..", "test", "ed_reference.jl"))
include(joinpath(@__DIR__, "..", "test", "ab_initio_fci_reference.jl"))

Random.seed!(42)

# H4 chain, STO-3G, near-equilibrium bond length. With the plain RHF trial
# and no force-bias sampling (Tier 1, see docs/afqmc_ab_initio_theory.md),
# this system's mild multi-reference character means AFQMC only recovers a
# modest fraction of the correlation energy -- shown here directly, not
# hidden. Compare against example/afqmc_example.jl's Hubbard case and the
# much tighter H2 agreement noted in the theory doc for how much trial
# quality matters.
bond = 1.4
mi = LatticeMC.build_h_chain_sto3g(bond; n_atoms=4)
trial = LatticeMC.build_rhf_trial(mi, 2, 2)

_, e_rhf = LatticeMC.rhf_scf(mi, 2)
e_fci = ab_initio_fci_ground_state_energy(mi.h1e, mi.h2e, 2, 2; E_nuc=mi.E_nuc)

result = LatticeMC.run_afqmc_ab_initio(mi, trial, 2, 2;
                                        dtau=0.01, num_walkers=300, num_steps=3000,
                                        equilibration_steps=500, stabilize_every=5,
                                        pop_control_every=10)

captured = 100 * (e_rhf - result.energy_mean) / (e_rhf - e_fci)
println("H4 chain, bond=$bond bohr, STO-3G:")
println("  E_RHF = $e_rhf")
println("  E_FCI = $e_fci")
println("  AFQMC = $(result.energy_mean) +/- $(result.energy_err)")
println("  correlation energy captured: $(round(captured, digits=1))%")

plt = plot(result.energy_trace, xlabel="Sample index (post-equilibration)", ylabel="Mixed-estimator energy",
           label="AFQMC", grid=true, dpi=800, title="H4 chain (STO-3G, bond=$bond)")
hline!(plt, [result.energy_mean], label="AFQMC mean", linestyle=:dash)
hline!(plt, [e_rhf], label="RHF", linestyle=:dot)
hline!(plt, [e_fci], label="FCI", linestyle=:dashdot)
savefig(plt, "afqmc_h4_trace.png")
display(plt)
