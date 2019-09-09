from setplanet import  SetPlanet
from planetarysystem import PlanetarySystem
import matplotlib.pyplot as plt
import numpy as np
import time
from utils import initial_walkers

# Planeta 1
Planet1 = SetPlanet("KOI1-b")
Planet1.boundaries(
    period=[18., 19], mass=[0, 100], 
    esinw=[-.5,.5], ecosw=[-.5, .5],
    mean_anomaly=[0., 360.],
    ascending_node=[179.63,179.63], inclination=[88., 92.])
ttvs1 = {}
f = np.genfromtxt("./example/Planet1.dat",skip_footer=0)
for i in f:
    ttvs1[int(i[0])] = [i[1], i[2], i[3] ]
Planet1.ttvs(ttvs1)




# Planeta 2
Planet2 = SetPlanet("KOI1-c")
Planet2.boundaries(
    ecosw=[-.5, .5], period=[42., 43.], 
    mass=[0, 100], mean_anomaly=[0., 360.],
    esinw=[-.5,.5], inclination=[88., 92.], 
    ascending_node=[170.,190.])
ttvs2 = {}
f = np.genfromtxt("./example/Planet2.dat",skip_footer=0)
for i in f:
    ttvs2[int(i[0])] = [i[1], i[2], i[3] ]
Planet2.ttvs(ttvs2)

 
# Create a Planetary System
KOI1 = PlanetarySystem( "KOI1", 1.0, 0.9, Ftime='Auto')

# Adding planets. The order is important

KOI1.add_planets([Planet1, Planet2])


# Run the optimizer
#RES_opt = KOI1.run_optimizers(cores=7, niter=28)

#print("\nRES:  ",RES_opt, "\n")
#RES_opt = [(1553692036.3067665, [91.33481315905478, 18.908817068432505, -0.31049618015937797, 88.66003819106518, -0.443397640907377, 199.4340940722509, 180.0, 43.97595500419743, 42.04772628463723, 0.38379242611121, 91.19455572548863, 0.11717768660317907, 13.999699213158124, 182.70349486948137]), (5252367485.329335, [27.441446904417603, 18.84540292201683, -0.4000289869531217, 89.9753662553519, -0.3931322936341178, 121.3094351682216, 180.0, 72.99921464741738, 42.02759616131987, 0.42558434615022905, 89.5796289002185, -0.00382601197924598, 24.01288978426942, 176.6051179684896]), (1887526915.3982801, [62.815783578710445, 18.914037372874276, -0.00634005092185097, 88.82779262807173, -0.16178187496639074, 197.69859006627655, 180.0, 59.97718937209646, 42.116312603782056, -0.15465374923446684, 90.3728495764764, -0.3174259788448167, 347.9317358256178, 179.43159430332295]), (1099796797.961756, [78.09805701435694, 18.777223866669615, -0.2866930726044331, 89.62955557130681, -0.3198140210031193, 199.73869443647484, 180.0, 96.86981936394713, 42.2475312449867, 0.45726334417239944, 89.78720002915713, 0.18916971154566498, 19.353413213795747, 188.1378094190082]), (1055054604.8520625, [87.25835025170385, 18.77823001471606, 0.2436982246406441, 89.14263032150983, 0.13948245329817152, 202.87612246691657, 180.0, 77.98936472518119, 42.16263132672816, 0.00187040140250061, 89.87981288975683, 0.1722672008538928, 354.97844352942195, 182.77838932091208]), (2379075744.95403, [75.1568999842174, 18.732200401002125, 0.13852360619765203, 88.4532924142072, 0.0012069845452700179, 216.57676774414153, 180.0, 52.436250600105105, 42.24075675440348, 0.15837017483261873, 89.00672702040805, 0.1218220265202764, 32.16163526706242, 177.14969544808537])]

#KOI1.plot_TTVs(RES_opt)
#plt.xlabel("algo")
#plt.savefig("ttvs_opt.png")
#plt.show()

#plt.hist([i[1][0] for i in RES_opt], label='0', alpha=0.3)
#plt.hist([i[1][7] for i in RES_opt], label='7', alpha=0.3)
#plt.legend()
#plt.show()


"""

solution = [39.57898,    18.74222,    -0.01863,    89.25392,
            0.48566,     326.26329,   180.0,       23.4167,
            42.42818,    -0.32348,    89.05876,    -0.33545,
            357.90397,   171.59549  ]

ti = time.time()

#SP = KOI1.run_TTVFast(solution)

chi2 = KOI1.calculate_chi2(solution)

print("Tiempo empleado: ", time.time() - ti)
print("Chi2: ", chi2)
"""


#Comienza MCMC

Ntemps = 7
Nwalkers = 64
Tmax = 1e6 



p0 = initial_walkers(KOI1, distribution="uniform",
                     ntemps=Ntemps, nwalkers=Nwalkers, 
                     opt_data=None, threshold=1.0)

print('tiempo ', KOI1.Ftime)

RES_mcmc = KOI1.run_mcmc(nwalkers=Nwalkers, Itmax=10000, conver_steps=50, 
                ntemps=Ntemps, Tmax=Tmax, betas=None,  bounds=KOI1.bounds, 
                pop0=p0, cores=7, suffix='')

print("hdf5: ", KOI1.hdf5_filename)

#KOI1.plot_hist(RES_mcmc.chain)
#plt.show()

"""
mass2 = RES_mcmc.chain[0,:,:,0].flatten()
mass1 = RES_mcmc.chain[0,:,:,7].flatten()
plt.hist(mass2, alpha=0.4, label='Mass2')
plt.hist(mass1, alpha=0.4, label='Mass1')
plt.legend()
plt.show()
"""

# Quisiera que fuera algo asi:
# KOI1.plotttvs
