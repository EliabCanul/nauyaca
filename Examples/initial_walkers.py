"""
initial_walkers.py

In example quick_fit.py we performed many runs of the optimizers and then we will use that information to initialize walkers in the MCMC. Here we illustrate the different strategies to initialize walkers.
"""


import nauyaca as nau
import matplotlib.pyplot as plt
import numpy as np

# Instead of defining again the Planetary System we used in quick_fit example, let's load the
# .json file containing all the information from that system.
PS = nau.PlanetarySystem.load_json('System-X.json')

# Now, let's call the results from the optimizers. You will realize that there are two
# files with suffix _cube and _phys. These correspond to the results in the normalized boundaries
# and in the physical values, respectively.

# Let's plot the physical results
physical = np.genfromtxt('System-X_phys.opt')

# Select a pair of parameters to visualize
# Change this parameters to generate new plots
x, y = 'ecc1', 'ecc2'

print("Valid parameters are: ")
print(PS.params_names.split())

# A dictionary with param names
dparam = dict(list(zip(PS.params_names.split(), range(PS.ndim))))


# Separate data from the file: chi2 and solutions
chi2 = physical.T[0]
phys = physical[:,1:].T

cx = phys[dparam[x]]
cy = phys[dparam[y]]

plt.figure(figsize=(15,10))

# Plot all the solutions from the optimizers
plt.scatter(cx, cy, 
	    c=np.log10(chi2), 
            cmap='inferno_r', 
	    label='Results from optimizers')
cb = plt.colorbar()
cb.set_label(r'$\log_{10} \chi^2$')

# Generate initial walkers, giving the normalized solutions
# Here you need to provide te NORMALIZED (cube) values.
# --> Try changing distribution and fbest!
wk = nau.utils.init_walkers(PS, 
			    opt_data= np.genfromtxt('System-X_cube.opt'),
                            ntemps=10, nwalkers=50, 
                            distribution='picked',  # Try: 'gaussian', 'picked', 'ladder'
                            fbest=.2)               # Change value between 0 and 1


# Convert initial walkers to physical  values
wk_phys = [nau.utils.cube_to_physical(PS, i) for i in wk.reshape(-1, wk.shape[-1]) ]
# Remove the constants added in the previous line
wk_phys = np.array([nau.utils._remove_constants(PS, i) for i in wk_phys]).T

# Plot the initial walkers belonging to all temperatures
plt.scatter(wk_phys[dparam[x]], 
	    wk_phys[dparam[y]], 
	    s=2, color='gray', alpha=0.5,
            label=f'Initial walkers')
plt.legend()
plt.xlabel(x)
plt.ylabel(y)
plt.savefig(f"optimizers_{x}_{y}.png")

# After generating the initial walkers 'wk' we can initialize the MCMC and setting the kwarg p0
"""
resmc = nau.MCMC(PS,
                tmax=100,
                itmax=500,         
                intra_steps=25,   
                cores=7,
		p0 = wk
                )

resmc.run()
"""

# An alternate way of running the MCMC without creating initial walkers explicitly is by providing 
# the necessary attributes to the MCMC class.
"""
resmc = nau.MCMC(PS,
                tmax=100,
                itmax=500,         
                intra_steps=25,   
                cores=7,
                opt_data=np.genfromtxt('System-X_cube.opt'),
                distribution='ladder',
                nwalkers=50,
                ntemps=10,
                fbest=0.2
                )

resmc.run()

"""


