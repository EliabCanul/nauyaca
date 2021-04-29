"""
restarting_mcmc.py

Sometimes the number of iterations is not enough to reach the convergence of the chains. We would like to restart the mcmc beginning from the last state of the chains from a previous mcmc run. For this example let's restart the mcmc done in the example 'quick_fit.py'. In that case the output file was System-X.hdf5
"""

import nauyaca as nau
import numpy as np

# Let's load the Planetary System object saved in the 'quick_fit' example
system = nau.PlanetarySystem.load_json('System-X.json')

# Restarting the mcmc using a new number of iterations and saving at each intra_steps
nau.MCMC.restart_mcmc(system, 
                hdf5_file=f'System-X.hdf5', 
                cores=8, 
                itmax=1000, 
                intra_steps=20, 
                restart_ladder=False # We specify that the temperature ladder continue as before
		)


# You will realize that a new .hdf5 file (and its corresponding .best file) is created with a suffix '_2'. Another suffix can be set through -suffix- kwarg.

print()
print()

# Now, let's join the chains stored in these two files

# First run
results1 = nau.utils.get_mcmc_results('System-X.hdf5', keywords=['CHAINS','NWALKERS'])
chains1 = results1['CHAINS'][0] # Choose the temperature 0
nw = results1['NWALKERS'][0] # the  number of walkers
print("number of walkers: ", nw)
print("chains1 shape: ", chains1.shape)

# Second run
results2 = nau.utils.get_mcmc_results('System-X_2.hdf5', keywords=['CHAINS'])
chains2 = results2['CHAINS'][0] # Choose the temperature 0
print("chains2 shape: ",chains2.shape)

# Joining
chains = np.concatenate([chains1, chains2], axis=1)
print("chains shape: ",chains.shape)


# The results from the mcmc are normalized between 0 and 1, which 
# correspond to the established boundaries. In order to get the physical 
# values it is necessary to convert them back.
# To convert from normalized values to physical, use nau.utils.cube_to_physical

chains_phys = np.array([list(nau.utils.cube_to_physical(system, x) for x in chains[w,:,:]) for w in range(nw) ])

# It will return the physical chains
print("chains physical shape: ", chains_phys.shape)


# Note that the shape of the physical chains have an extra dimension. 
# It is because nau.utils.cube_to_physical inserts the constant parameters
# stored in system.constant_params
print("Constant parameters inserted: ", system.constant_params)
# which corresponds to the index and the value


# If you need the physical values without the constant parameters,
# you can remove them with the function _remove_constants in utils
chains_phys = np.array([list(nau.utils._remove_constants(system, x) for x in chains_phys[w,:,:]) for w in range(nw) ])
print("chains physical shape (no constants): ", chains_phys.shape)


# Now, do whatever operation with these chains. Remember that the dimensions are
print()
print("physical dimensions: ", system.params_names)

# Results for mass of planet 1
mass1 = np.mean(chains_phys[:,:,0].flatten())
mass_std = np.std(chains_phys[:,:,0].flatten())
print(f"Mass 1: {mass1} +/- {mass_std}")


