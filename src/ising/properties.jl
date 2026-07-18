
function magnetization(spin_array::IsingModel, skipped_points::Int64, total_points::Int64, T::Float64, k::Float64=1.0)
    """
    parameters:
    spin_array: IsingModel object
    skiped_points: number of points to skip
    total_points: total number of points
    T: temperature
    k: Boltzmann constant
    _________________________
    returns:
    ms_j: mean magnetization
    ms_squared_j: mean squared magnetization
    """

    J = spin_array.J
    bj = J / (k * T)
    lattice = spin_array.lattice
    N = spin_array.N

    spin, energies, spins = metropolis(spin_array, total_points, bj, energy_manual(spin_array))
    ms_j = abs(mean(spin[skipped_points:N*10:end]))
    ms_squared_j = mean(spin[skipped_points:N*10:end] .^ 2)
    return ms_j, ms_squared_j
end

function magnetic_susceptibility(spin_array::IsingModel, skipped_points::Int64, total_points::Int64, T::Float64)
    """
    parameters:
    spin_array: IsingModel object
    skiped_points: number of points to skip
    total_points: total number of points
    T: temperature
    _________________________
    returns:
    chi: magnetic susceptibility
    """
    Ms, Ms_squared = magnetization(spin_array, skipped_points, total_points, T)
    N = spin_array.N
    k = 1.0
    bj = spin_array.J / (k * T)
    T_c = 2.269 * spin_array.J / k
    if T > T_c
        chi = (Ms_squared) * (N^2) / (k * T)
    elseif T < T_c
        chi = (Ms_squared - Ms.^2) * (N^2) / (k * T)
    end

    return chi
end

function standard_deviation_spin(spin_array::IsingModel, skipped_points::Int64, total_points::Int64, T::Float64) 
    """
    parameters:
    spin_array: IsingModel object
    skiped_points: number of points to skip
    total_points: total number of points
    T: temperature
    _________________________
    returns:
    sigma_m: standard deviation of magnetization

    """
    Ms, Ms_squared = magnetization(spin_array, skipped_points, total_points, T)
    return sqrt(Ms_squared - Ms^2)
end
"""
# Heat Capacity
Heat capacity, C_v=frac{sigma^2_E}{T^2}
=(<E^2>-<E>^2).beta^2k^2
=sigma_{E/J}^2.(beta J)^2k^2
"""
function specific_heat(spin_array::IsingModel, skipped_points::Int64, total_points::Int64, T::Float64)
    """
    parameters:
    spin_array: IsingModel object
    skiped_points: number of points to skip
    total_points: total number of points
    T: temperature
    _________________________
    returns:
    C_v: specific heat
    """
    N = spin_array.N
    k = 1.0
    bj = spin_array.J / (k * T)
    E_mean = mean(energies[skipped_points:end])
    E_squared_mean = mean(energies[skipped_points:end] .^ 2)
    return (E_squared_mean - E_mean^2) * (k * bj^2)

end

function cumulant_spin(spin_array::IsingModel, skipped_points::Int64, total_points::Int64, T::Float64)
    """
    parameters:
    ms_squared_avg: mean squared magnetization
    ms_avg: mean magnetization
    _________________________
    returns:
    U: cumulant
    """
    N = spin_array.N
    k = 1.0
    bj = spin_array.J / (k * T)
    spin, energies, spins = metropolis(spin_array, total_points, bj, energy_manual(spin_array))
    ms_squared_j = mean(spin[skipped_points:N*10:end] .^ 2)
    m_4 = mean(spin[skipped_points:N*10:end] .^ 4)
    return 1.0 - m_4 / (3.0 * ms_squared_j^2)
end

function autocorrelation_montecarlo(spin_array::IsingModel, total_points::Int64, l::Int64, T::Float64)
    """
    parameters:
    spin_array: IsingModel object
    total_points: total number of points
    l: lag parameter for autocorrelation
    T: temperature
    _________________________
    returns:
    autocorrelation: autocorrelation
    """
    numerator = 0.0
    denominator = 0.0
    k = 1.0
    bj = spin_array.J / (k * T)
    spin, energies, spins = metropolis(spin_array, total_points, bj, energy_manual(spin_array))
    ms_j = abs(mean(spin))
    # display(ms_j)
    for i in 1:total_points - l-1
        numerator += abs(spin[i]) * abs(spin[i + l]) - ms_j^2
    end
    for i in 1:total_points-1
        denominator += abs(spin[i]) * abs(spin[i]) - ms_j^2
    end
    autocorrelation = numerator / denominator
    # display(autocorrelation)
    return autocorrelation
end

# temperatures = [1.6, 2.0, 2.16, 2.34, 2.43, 2.76, 3.35]
# skipped_points = 10000
# total_points = 200000
# total_points = 400000
# l=150
# J=1.0
# lattice_n=IsingModel(5,J, false)  
# autocorr=[]
# plt = plot(size=(800, 500), legend=:topright, xlabel="l", ylabel="autocorrelation", grid=false)
# autocorrelation_values = autocorrelation_montecarlo(lattice_n,total_points,5,1.0)
# for i in 1:l
#     autocorrelation_values = autocorrelation_montecarlo(lattice_n,total_points,i,1.0)
#     push!(autocorr,autocorrelation_values)
# end
# display(autocorr)
# plot!(autocorr,label="N=50",linewidth=2,marker=:auto)
# plot()
# ms_values = Float64[]
# chi_values = Float64[]
# for temperature in temperatures
#     spin_array = IsingModel(50)
#     chi = magnetic_susceptibility(spin_array, skipped_points, total_points, temperature)
#     ms, ms_squared = magnetization(spin_array, skipped_points, total_points, temperature)
#     display(ms)
#     push!(ms_values, ms)
#     push!(chi_values, chi)
# end

# # Plot results for each lattice size
# plot!(temperatures, chi_values, label="L = 50")
