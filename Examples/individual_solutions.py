"""
individual_solutions.py


"""

import nauyaca as nau
import numpy as np
import matplotlib.pyplot as plt

# Let's load the Planetary System object
mysystem = nau.PlanetarySystem.load_json('System-X.json')
print(mysystem)

# Now, let's use many of the functions in the utils module.

# Mid-transit times are simulated from a given initial conditions
# Then, these transits are compared with the observations. 
# Let's calculate the chi square statistic of random solutions.

# Let's read the mcmc chains from a previous run
results = nau.utils.get_mcmc_results('./System-X.hdf5', keywords=['CHAINS'])
chains = results['CHAINS'][0]
nw, steps, _ = chains.shape

# Now, we select a random solution
rdm_wk = np.random.randint(0,nw)
# Take a random step after second half of the iterations
# in order to get a more converged solution
rdm_st = np.random.randint(0.5*steps,steps)
print()
print(f"Random solutions|  walker:{rdm_wk}  step:{rdm_st}")

# There are two ways of calculating the chi square:
# 1.- From the normalized solution
# 2.- From the physical solution 

print("1.------------------------------")
# Extract that random solution
x_cube = chains[rdm_wk, rdm_st, :]
print("Normalized solution:\n", x_cube)

# Let's pass the proposal and the Planetary System object
chi2 = nau.utils.calculate_chi2(x_cube, mysystem)
print("chi2: ", chi2)
print()

print("2.------------------------------")
# Convert the normalized solution to physical values
x_phys = nau.utils.cube_to_physical(mysystem, x_cube)
print("Physical solution:\n", x_phys)


# An easy way to know what that solution means:
print("Dictionary: ")
print(dict(zip(mysystem.params_names_all.split(), x_phys)))

# Now, we use the function calculate_chi2_physical
chi2 = nau.utils.calculate_chi2_physical(x_phys, mysystem)
print("chi2: ",chi2)
print()

print("Extra: ------------------------------")
# There are many other functionalities in calculate_chi2_physical
# For example, we can get the individual chi2 ṕer planet
chi2 = nau.utils.calculate_chi2_physical(x_phys, mysystem, individual=True)
print("Individual chi2:",chi2)
# which  returns a dictionary with the individual chi squares per planet in the system


# We can also get the simulated ephemeris of these planets
chi2, ephe = nau.utils.calculate_chi2_physical(x_phys, mysystem, get_ephemeris=True)
print()
print("chi2:", chi2)
print("Simulated ephemeris:\n",ephe)


# It returns a tuple with the chi2 and a dictionary with the simulated ephemeris per
# planet, where keys are the transit epoch number and values are the mid-trantis times


# Want to know how that solution looks like?
# Use the Plots module in nauyaca

naup = nau.Plots(mysystem)

# Remove the constant parameters. They will be automatically
# added in TTVs plot function
x_phys_2 = nau.utils._remove_constants(mysystem, x_phys)

naup.TTVs(flat_params=x_phys_2 )


plt.show()











