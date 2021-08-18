"""
custom_fit.py

Unlike the quick_fit example, here we show extra features in nauyaca to adapt the simulations to your requirements. It includes parcial simulations of TTVs ephemeris, saving at specific directories and generate figures for assesing the MCMC performance.
"""
import nauyaca as nau
import numpy as np
import matplotlib.pyplot as plt

directory = "./inputs/"

P1 = nau.SetPlanet('Planet-b')
P1.mass = [1,50]
P1.period = [34.5,34.6]
P1.ecc = [0.0,0.1]
P1.inclination = [85,95]
P1.ascending_node = [88.79,88.79]
P1.load_ttvs(directory + "3pl_planet0_ttvs.dat")

P2 = nau.SetPlanet('Planet-c')
P2.mass = [1,50]
P2.period = [66.0,66.1]
P2.ecc = [0.0,0.1]
P2.inclination = [85,95]
P2.ascending_node = [0,180]
P2.load_ttvs(directory + "3pl_planet1_ttvs.dat")

P3 = nau.SetPlanet('Planet-d')
P3.mass = [1,150]
P3.period = [125.8,125.9]
P3.ecc = [0.0,0.1]
P3.inclination = [85,95]
P3.ascending_node = [0,180]
P3.load_ttvs(directory + "3pl_planet2_ttvs.dat")

PS = nau.PlanetarySystem("MySystem", mstar=1.08, rstar=1.0)
PS.add_planets([P1,P2,P3])

# Print the TTVs
print(PS.TTVs)

# From  the output above it is seen that the observations span
# around 4500 days. Setting ftime = 800 will discard all the observations
# greater than 800 days. It is useful when you want to quickly explore
# the possible solutions without evaluating all the transits. 
PS.simulation(t0=0, dt=1.1, ftime=800)
print(PS)

# This file is saved in the working directory
PS.save_json

# Change the directory and add a suffix to the outputs
optim = nau.Optimizers(PS, nsols=84, cores=7,
			path = "./outputs/",
			suffix = "_3pl"			
			)

opt_solutions = optim.run()


# unlike the example
# By default, the hdf5 file is the planetary system name, but you can change it.
resmc = nau.MCMC(PS,
                tmax=100,
                itmax=1000,         
                intra_steps=20,   
                cores=7,
		opt_data=np.genfromtxt('./outputs/MySystem_cube_3pl.opt'),
                distribution='gaussian',
                nwalkers=50,
                ntemps=10,
                fbest=0.2,
		path='./outputs/',
		file_name='Custom_hdf5',
		suffix='_mcmc'
                )

resmc.run()


# Let's consider all the chains so burnin=0
nauplot = nau.Plots(PS, hdf5_file=resmc.hdf5_filename, burnin=0)

print()

nauplot.monitor(size=(15,10))
plt.savefig('cf_monitor.jpg')


nauplot.convergence(nchunks_gr=15)
plt.savefig('cf_convergence.jpg')




