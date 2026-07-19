module LatticeMC

module Ising

using LinearAlgebra
using Random
using Printf
using Statistics
using Plots

export IsingModel, display_lattice, energy_manual, energy_periodic,
       periodic_boundary_conditions, periodic_boundary_conditions_manual,
       create_periodic_boundary_lookup_table, periodic_boundary_conditions_lookup,
       calculate_probability, create_probability_lookup_table, calculate_probability_lookup,
       metropolis, get_spin_energy,
       magnetization, magnetic_susceptibility, standard_deviation_spin,
       specific_heat, cumulant_spin, autocorrelation_montecarlo

include("ising/Ising_model.jl")
include("ising/metropolis.jl")
include("ising/properties.jl")

end # module Ising

module AFQMC

using LinearAlgebra
using Random
using Statistics

export HubbardLattice, build_hubbard_chain, build_hubbard_square,
       TrialWavefunction, build_trial_wavefunction,
       build_uhf_trial, bipartite_coloring,
       Walker, init_walkers, orthonormalize!,
       greens_function, overlap, walker_overlap, local_energy, mixed_energy_estimator,
       local_update_ratio, rank1_update_greens_function!,
       build_expK_half, hs_gamma, propagate_step!,
       population_control!,
       population_control_signed!, propagate_step_free!, mean_sign,
       run_afqmc, block_average,
       MolecularIntegrals, read_fcidump,
       build_h_chain_sto3g,
       cholesky_decompose_eri, modified_one_body,
       AbstractAbInitioTrial, AbInitioTrial, build_rhf_trial, rhf_scf,
       build_uhf_trial_ab_initio, uhf_scf,
       MultiDetTrial, multidet_from_ci, build_casci_trial, determinant_matrix, mo_transform,
       AbInitioWalker, init_ab_initio_walkers, orthonormalize_ab_initio!,
       greens_function_ab_initio, overlap_ab_initio, walker_overlap_ab_initio,
       local_energy_ab_initio, mixed_energy_estimator_ab_initio,
       propagate_step_ab_initio!,
       force_bias_shift, propagate_step_ab_initio_force_bias!,
       population_control_ab_initio!,
       run_afqmc_ab_initio

include("afqmc/lattice.jl")
include("afqmc/trial_wavefunction.jl")
include("afqmc/uhf_trial.jl")
include("afqmc/walker.jl")
include("afqmc/estimators.jl")
include("afqmc/propagator.jl")
include("afqmc/population_control.jl")
include("afqmc/driver.jl")

module AbInitio

using LinearAlgebra
using Random
using Statistics
using SpecialFunctions
import ..block_average

export MolecularIntegrals, read_fcidump,
       build_h_chain_sto3g,
       cholesky_decompose_eri, modified_one_body,
       AbstractAbInitioTrial, AbInitioTrial, build_rhf_trial, rhf_scf,
       build_uhf_trial_ab_initio, uhf_scf,
       MultiDetTrial, multidet_from_ci, build_casci_trial, determinant_matrix, mo_transform,
       AbInitioWalker, init_ab_initio_walkers, orthonormalize_ab_initio!,
       greens_function_ab_initio, overlap_ab_initio, walker_overlap_ab_initio,
       local_energy_ab_initio, mixed_energy_estimator_ab_initio,
       propagate_step_ab_initio!,
       force_bias_shift, propagate_step_ab_initio_force_bias!,
       population_control_ab_initio!,
       run_afqmc_ab_initio

include("afqmc/ab_initio/integrals.jl")
include("afqmc/ab_initio/sto3g_hydrogens.jl")
include("afqmc/ab_initio/cholesky.jl")
include("afqmc/ab_initio/rhf_trial.jl")
include("afqmc/ab_initio/uhf_trial.jl")
include("afqmc/ab_initio/walker.jl")
include("afqmc/ab_initio/estimators.jl")
include("afqmc/ab_initio/multidet_trial.jl")
include("afqmc/ab_initio/casci_trial.jl")
include("afqmc/ab_initio/propagator.jl")
include("afqmc/ab_initio/population_control.jl")
include("afqmc/ab_initio/driver.jl")

end # module AbInitio

using .AbInitio

end # module AFQMC

module Heisenberg

using Random

export HeisenbergLattice, build_heisenberg_chain, build_heisenberg_square,
       nbonds, is_bipartite, run_sse

include("heisenberg/lattice.jl")
include("heisenberg/sse.jl")

end # module Heisenberg

using .Ising
using .AFQMC
using .Heisenberg

end # module LatticeMC
