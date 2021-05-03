"""
quick_fit.py

In this example we consider the inversion problem of two planets. 
We show the standard way for using nauyaca in order to estimate the planet masses and orbits. 
This is a synthetic planetary system whose transit ephemeris have been calculated as 
described in Canul et al. (2021). We obtain finally the MCMC posteriors and make a pair of 
figures to evaluate our results.
"""

import nauyaca as nau
import matplotlib.pyplot as plt
import numpy as np

directory = "./inputs/"

# Defining planet objects: Names and boundaries
# Planet 1
P1 = nau.SetPlanet('Planet-b')
P1.mass = [1,100]
P1.period = [33.61,33.62]
P1.ecc = [0.0,0.3]
P1.inclination = [85,95]
P1.ascending_node = [179.99,179.99]  # Keep the ascending node of one planet as fixed
P1.load_ttvs(directory + "planetb_ttvs.dat")

# Planet 2
P2 = nau.SetPlanet('Planet-c')
P2.mass = [1,100]
P2.period = [73.5,73.51]
P2.ecc = [0.0,0.3]
P2.inclination = [85,95]
P2.ascending_node = [90,270] # Consider prograde solutions 
P2.load_ttvs(directory + "planetc_ttvs.dat")

# Creating Planetary System object
PS = nau.PlanetarySystem("System-X", mstar=0.91, rstar=1.18)
PS.add_planets([P1,P2])

# Simulation attributes.
# Here, t0 is the reference time of the osculating elements
PS.simulation(t0=0, dt=1.1)

# Print a summary of the Planetary System
print(PS)

# Save the Planetary System for further examples
# in json format
PS.save_json


# Running optimizers. Increase the number of solutions to reach
optim = nau.Optimizers(PS, nsols=80, cores=8)
opt_solutions = optim.run()


# Running the MCMC. Try increasing the number of iterations, walkers or temperatures
resmc = nau.MCMC(PS,
                tmax=100,    # Maximum temperature in ladder (see ptemcee documentation)
                itmax=5000,   # Maximum number of iterations      
                intra_steps=20,   # thinning factor
                cores=7,   # Cores to run in parallel
                opt_data=opt_solutions,  # Solutions from optimizers
                distribution='ladder',  # A strategy to initialize walkers
                nwalkers=50,  # Number of walkers
                ntemps=10,  # Number of temperatures
                fbest=0.2   # a fraction of the best optimizer results
                )
resmc.run()

# Print a summary of the results and get a dictionary with the posteriors:
posteriors = nau.utils.mcmc_summary(PS, hdf5_file=resmc.hdf5_filename,
				    get_posteriors=True)


# Plot a pair of results
x_post = 'mass1'
y_post = 'mass2'
plt.scatter(posteriors[x_post], posteriors[y_post])
plt.xlabel(x_post)
plt.ylabel(y_post)
plt.savefig(f'qf_{x_post}_{y_post}.jpg')

# See the available keys in posteriors
print("\nAvailable posteriors are:")
print(list(posteriors.keys()))
print()

# Plot a pair of built-in figures from the nauyaca's Plots module.
# burnin will discard the first 25% of the initial chains
nauplot = nau.Plots(PS, hdf5_file=resmc.hdf5_filename, burnin=0.25)

# Plot the fitted TTVs
nauplot.TTVs(nsols=20, mode='random')
plt.savefig('qf_TTVs.jpg')

# Plot the chains
nauplot.trace_plot()
plt.savefig('qf_chains.jpg')

# Plot histograms of the posteriors
nauplot.hist(titles=True)
plt.savefig('qf_hist.jpg')


"""
Compare your results! The true planetary parameters for this
synthetic system are:

{'mass1': 9.05014,
 'period1': 33.61783,
 'ecc1': 0.0968,
 'inclination1': 89.87,
 'argument1': 333.3122,
 'mean_anomaly1': 235.68,
 'ascending_node1': 179.99,
 'mass2': 17.89135,
 'period2': 73.50794,
 'ecc2': 0.0957,
 'inclination2': 89.87,
 'argument2': 79.0444,
 'mean_anomaly2': 10.27,
 'ascending_node2': 180.0}
"""
