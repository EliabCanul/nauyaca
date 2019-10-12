from setplanet import  SetPlanet
from planetarysystem import PlanetarySystem
from operatettvs import Optimizers, MCMC
from plots import *

from utils import initial_walkers

import matplotlib.pyplot as plt
import numpy as np
import pickle


# ------------  Planeta 1
Planet1 = SetPlanet("KOI1-b")
Planet1.set_boundaries(
    mass=[.1, 50],
    period=[18.8, 18.9],  
    ecc=[0., 0.5], 
    inclination=[91.26, 91.26],
    argument=[1., 360.], 
    mean_anomaly=[0, 360.], 
    ascending_node=[179.63, 179.63])
ttvs1 = {}
f = np.genfromtxt("./example/Planet1.dat")
for i in f:
    ttvs1[int(i[0])] = [i[1], i[2], i[3] ]
Planet1.add_ttvs(ttvs1)


# ------------ Planeta 2
Planet2 = SetPlanet("KOI1-c")
Planet2.set_boundaries( 
    mass=[.1, 50], 
    period=[42.2, 42.3],
    ecc=[0, .5],  
    inclination=[89.34, 89.34],
    argument=[0.1, 360.], 
    mean_anomaly=[0., 360.],
    ascending_node=[160., 190.])
ttvs2 = {}
f = np.genfromtxt("./example/Planet2.dat")
for i in f:
    ttvs2[int(i[0])] = [i[1], i[2], i[3] ]
Planet2.add_ttvs(ttvs2)

 
# Create a Planetary System
KOI1 = PlanetarySystem( "KOI1", 1.0, 1.0, Ftime='default')

# Adding planets. Add planets in a list. The order is important
KOI1.add_planets([Planet1, Planet2])



# ====== Run the optimizer

RES_opt = Optimizers.run_optimizers(KOI1, cores=7, niter=7)

print('RES: ', RES_opt)

sort_res = sorted(RES_opt, key=lambda j: j[0])
n_sols = 3
best_3_solutions = [s[1:] for s in sort_res[:n_sols]]
Plots.plot_TTVs(KOI1, best_3_solutions)
plt.show()



# Comienza MCMC
Ntemps = 14
Nwalkers = 64
Tmax = None

p0 = initial_walkers(KOI1, distribution="Picked", 
                     ntemps=Ntemps, nwalkers=Nwalkers, 
                     opt_data=RES_opt, threshold=0.5)



RES_mcmc = MCMC.run_mcmc(KOI1, nwalkers=Nwalkers, Itmax=500, conver_steps=50, 
                ntemps=Ntemps, Tmax=Tmax, betas=None,
                pop0=p0, cores=7, suffix='')
