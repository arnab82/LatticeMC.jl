using LatticeMC
using JLD2

lattice_n = LatticeMC.IsingModel(50, 1.0, true)
display(lattice_n)
enrg = LatticeMC.energy_manual(lattice_n)
display(enrg)
@save "lattice_n.jld2" lattice_n enrg