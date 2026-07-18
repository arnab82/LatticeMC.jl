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
       run_afqmc, block_average

include("afqmc/lattice.jl")
include("afqmc/trial_wavefunction.jl")
include("afqmc/uhf_trial.jl")
include("afqmc/walker.jl")
include("afqmc/estimators.jl")
include("afqmc/propagator.jl")
include("afqmc/population_control.jl")
include("afqmc/driver.jl")

end # module AFQMC

using .Ising
using .AFQMC

end # module LatticeMC
