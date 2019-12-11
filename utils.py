import numpy as np
import ttvfast
import sys
import h5py

# Helpful variables
Returns = {"OPT1": np.inf, "OPT2": 1e50, "OPT3": 1.0,
            "MCMC1": -np.inf, "MCMC2": -1e50, "MCMC3": -1.0}

Mearth_to_Msun = 3.0034893488507934e-06


def run_TTVFast(flat_params, mstar=None, NPLA=None, Tin=0., Ftime=None):
    """A wrapper to run TTVFast
    
    Arguments:
        flat_params {Array} -- A flat array containing mass, period, eccentricity,
            inclination, argument, mean anomaly, ascending node for all planets
    
    Keyword Arguments:
        mstar {float} -- stellar mass in solar masses (default: {None})
        NPLA {int} -- Number of planets in the system (default: {None})
        Tin {float} -- Initial time for simulation (default: {0.})
        Ftime {float} -- Final time of the simulation (default: {None})
    
    Returns:
        Array -- An array 'SP' with transit numbers and transit epochs for all 
        planets labeled in the entrance order from flat_params. 
        From TTVFast doc:
        [PLANET,  EPOCH,  TIME (DAYS),  RSKY (AU),  VSKY (AU/DAY)]
        SP[0] = planet index, SP[1] = transit number, SP[2] = central time, 
        SP[3] = Rsky, SP[4] = Vsky
    """

    # Split 'flat_params' using 7 parameters per planet
    iters = [iter(flat_params)] * 7
    planets = list(zip(*iters))

    # TODO: Verify that planets contains exactly 7 parameters per planet

    # Iteratively adds planet's parameters to TTVFast
    planets_list = []
    min_period = min([ p[1] for p in planets ])

    for _, planet in enumerate(planets):
        # Be careful with the order!: m, per, e, inc, omega, M, Omega
        # See angle definitions: 
        # https://rebound.readthedocs.io/en/latest/python_api.html
        # TODO: refer to Nauyaca documentation (when exist)
        
        planets_list.append(
        ttvfast.models.Planet( 
            mass= planet[0]*Mearth_to_Msun , #*u.Mearth).to(u.Msun).value ,
            period = planet[1], 
            eccentricity = planet[2],
            inclination = planet[3],
            argument = planet[4],
            mean_anomaly = planet[5],
            longnode = planet[6] 
            )
            ) 

    signal = ttvfast.ttvfast(
            planets_list, stellar_mass=mstar, time=Tin, dt=min_period/100., 
            total=Ftime, rv_times=None, input_flag=1)   
    
    SP = signal['positions']

    try:
        lastidx = SP[2].index(-2.0) # -2.0 is indicatiive of empty data
        SP = [ i[:lastidx] for i in SP] # Remove array without data
    except:
        pass
    
    return SP


def calculate_epochs(SP, self):

    """ SP: signal position array from TTVFast
        Signal Positions list contains:
        [PLANET,  EPOCH,  TIME (DAYS),  RSKY (AU),  VSKY (AU/DAY)]
        SP[0] = planet index, SP[1] = transit number, SP[2] = central time, 
        SP[3] = Rsky, SP[4] = Vsky"""
    
    EPOCHS = {}
    # Identify the first transit planet in the simulation
    try:
        # Find the array index of the first transit of the 
        # observed first_planet_transit
        refpla = SP[0].index(self.planets_IDs[self.first_planet_transit]) 
        
        # Simulated time where the first transit of first_planet_transit 
        # occurs.
        t0_simulation = SP[2][refpla]

        # Save the transit epochs of every planet and convert them 
        # to julian days. Time 0 for the simulations in julian days. 
        # This is the reference epoch for the whole simulations
        T0 = self.T0JD - t0_simulation  
        for k, v in self.planets_IDs.items(): 
            EPOCHS[k] = {item[1]:T0+item[2] for item in list(zip(*SP)) 
                        if item[0]==v and item[3]<= self.rstarAU}
    except:
        #pass
        print('Warning: Invalid proposal')
        
    return EPOCHS    


def calculate_chi2(flat_params, self, flag="OPT"):
    """Calculate Chi square statistic between simulated transit times and
    observed.
    
    Arguments:
        flat_params {Array} -- A flat array containing mass, period, eccentricity,
            inclination, argument, mean anomaly, ascending node for all planets
    
    Keyword Arguments:
        flag {str} -- A flag to switch between minimum or maxima searching.
        Set 'OPT' if you are looking for minima, as in optimizers.
        Set 'MCMC' if you are looking for maxima, as in mcmc.
        (default: {"OPT"})
    
    Returns:
        float -- Returns chi square for 'OPT' option. Returns minus chi square
            for 'MCMC' option.
    """

    # Verify that proposal is inside boundaries
    inside_bounds = intervals(self.bounds, flat_params) 
    if False in inside_bounds:
        return Returns[flag+"1"] 
    else:
        pass

    # Reconstruct flat_params adding the constant values
    flat_params = list(flat_params)
    for k, v in sorted(self.constant_params.items(), key=lambda j:j[0]):
        flat_params.insert(k, v)

    # Get 'positions' from signal in TTVFast
    signal_pos = run_TTVFast(
               flat_params,  mstar=self.mstar, NPLA=self.NPLA, 
               Ftime=self.sim_interval )

    # Calculate transits epochs
    EPOCHS = calculate_epochs(signal_pos, self)

    # Compute chi^2
    try:
        chi2 = 0.0
        for plnt_id, ttvs_obs in self.TTVs.items():
            ttvs_sim = EPOCHS[plnt_id]
            for epoch, times in ttvs_obs.items():
                sig_obs = (times[1] + times[2])/2.
                chi2 = chi2 + ((times[0] - ttvs_sim[epoch])/sig_obs)**2

        return Returns[flag+"3"] * chi2
    except:
        return Returns[flag+"2"]


# -----------------------------------------------------
# ADITIONAL FUNCTIONS TO MAKE LIFE EASIER

def initial_walkers(self, ntemps=None, nwalkers=None, distribution=None,
                    opt_data=None, threshold=1.0):
    """An useful function to easily create initial walkers.
    
    Keyword Arguments:
        ntemps {int} -- Number of temperatures (default: {None})
        nwalkers {int} -- number of walkers (default: {None})
        distribution {str} -- A string option from:  {'Uniform' | 'Gaussian' | 'Picked'}
        opt_data {str} -- Data from optimizers. Just used if Uniform or Picked
            options are selected (default: {None})
        threshold {float} -- A value between 0 and 1 to select the fraction of 
            solutions from opt_data to take into account.
            For example: threshold=0.5 takes the best half of solutions from 
            opt_dat (default: {1.0})
    
    Returns:
        Array -- An array of shape (ntemps, nwalkers, dimension)
    """

    assert(0.0 <= threshold <= 1.0), f"threshold must be between 0 and 1!"

    if nwalkers < 2*self.NPLA*7:
        sys.exit("Number of walkers must be >= 2*ndim, i.e., \n \
            nwalkers >= {}.\n Stopped simulation!".format(2*self.NPLA*7))

    if distribution.lower() == 'uniform':
        return func_uniform(self, ntemps=ntemps, nwalkers=nwalkers)

    if distribution.lower() in ("gaussian", "picked") and opt_data is not None:
        return func_gp(self, distribution, ntemps=ntemps, nwalkers=nwalkers,
                    opt_data=opt_data, threshold=threshold)
    
    else:
        text = ("--> Argument \'distribution\' does not match with any",
        " supported distributions. Available options are:",
        "\n *Uniform \n *Gaussian \n *Picked ",
        "\n For the last two, please provide results from the ",
        "optimizer routine.")
        sys.exit(" ".join(text))

        
def func_uniform(self, ntemps=None, nwalkers=None):
    print("\n--> Selected distribution: Uniform")
    POP0 = []
    for i in self.bounds:
        linf = i[0]
        lsup = i[1]

        # Create uniform random walkers between boundaries
        RDM = np.random.uniform(linf , lsup, ntemps*nwalkers)
        POP0.append(RDM)
    POP0 = np.array(POP0).T.reshape(ntemps, nwalkers, len(self.bounds))  # !!!!

    return POP0


def func_gp(self, distribution, ntemps=None, nwalkers=None, 
            opt_data=None, threshold=1.0):

    print("\n--> Selected distribution: {}".format(distribution))

    def random_choice(distr, par):
        """Returns a value form param according to the chosen distribution
        
        Arguments:
            distr {str} -- the chosen distribution
            par {array} -- an array of any parameter to sample
        
        Returns:
            float -- a random value given by the distribution
        """
        if distr.lower() == 'gaussian':
            mu = np.mean(par)
            sig = np.std(par)

            if sig == 0.0:
                return mu
            else:
                return np.random.normal(loc=mu,scale=sig)
        
        if distr.lower() == 'picked':
            return np.random.choice(par)

    # Clean and sort results. Get just data inside threshold.
    [opt_data.remove(res) for res in opt_data if res[0]==1e50]
    opt_data = sorted(opt_data, key=lambda x: x[0])
    cut = int(len(opt_data)*threshold) - 1 # index of maximum chi2
    opt_data = opt_data[:cut]
    params = np.array([x[1:] for x in opt_data]).T
    
    POP0 = []
    i = 0
    for par_idx, param in enumerate(params):
        # Initialize walkers avoiding the constant paremeters
        if par_idx not in list(self.constant_params.keys()):
            poptmp = [] 
            while len(poptmp) < ntemps*nwalkers:
                rdm = random_choice(distribution, param)
                if self.bounds[i][0] <= rdm and rdm <= self.bounds[i][1]:
                    poptmp.append(rdm)

            POP0.append(poptmp)
            i += 1
    POP0 = np.array(POP0).T.reshape(ntemps, nwalkers, len(self.bounds))
    
    return POP0

def mcmc_summary(hdf5_file, burning=0.0):

    assert(0.0 <= burning <= 1.0), f"burning must be between 0 and 1!"

    #output_name= hdf5_file.split()[0]
    f = h5py.File(hdf5_file, 'r')
    syst_name = f['NAME'].value
    colnames = f['COL_NAMES'].value[:]
    
    index = f['INDEX'].value[0]
    
    ms = f["MSTAR"].value[0]
    rs = f["RSTAR"].value[0]
    npla = f['NPLA'].value[0]
    
    nw = f["NWALKERS"].value[0]
    nt = f["NTEMPS"].value[0]
    cs = f["CONVER_STEPS"].value[0]
    
    maxc2 = f["BESTCHI2"].value[:index+1]
    bs = f["BESTSOLS"].value[:index+1]
    
    burning = int(burning*index)
    chains = f['CHAINS'].value[0,:,burning:index+1,:] #Just for temperature 0

    f.close()
    
    best = zip(maxc2, bs)

    # Reverse the list because we sorte by -chi**2
    sort_res = sorted(list(best), key=lambda j: j[0], reverse=True)
    best_chi2, best_sol = sort_res[0][0], sort_res[0][1]

    
    #chains = chains[:,burning:index+1,:]
    
    print("-->Planetary System: ", syst_name)
    print("   Stellar mass: ", ms)
    print("   Stellar radius: ", rs)
    print("   Number of planets: ", npla)
    print("--------------------------")
    print("-->MCMC parameters")
    print("   Ntemps: ", nt)
    print("   Nwalkers per temperature: ", nw)
    print("   Thining: ", cs)
    print("--------------------------")
    print("      RESULTS             ")
    print("-->Best solution in MCMC")
    print("   Best chi2 solution: ", round(best_chi2,5))
    for i in range(npla):
        print("   " + "   ".join( str(round(k,4)) for k in np.array_split(best_sol, npla)[i]) )
    print("--------------------------")    
    print("-->MCMC medians and 1-sigma errors")

    for i, name in enumerate(list(colnames.split())):
        parameter = chains[:,:,i].flatten()

        low, med, up = np.percentile(parameter, [16,50,84])

        if i == 1: 
            # For period increase decimals
            tit = "%s ^{+%s}_{-%s} " % (round(med,4),
                                            round(up-med,4),
                                            round(med-low,4))
        elif i == 2:
            # For eccentricity increase decimals
            tit = "%s ^{+%s}_{-%s}" % (round(med,3),
                                            round(up-med,3),
                                            round(med-low,3))
        else:
            tit = "%s ^{+%s}_{-%s}" % (round(med,2),
                                            round(up-med,2),
                                            round(med-low,2))
        #ndim += 1
        print("   %15s      %20s" % (name, tit))
    print("--------------------------") 
    
    return

def geweke(self, hdf5_file=None, burning=0.0):
    # Geweke criterion
    # https://rlhick.people.wm.edu/stories/bayesian_5.html
    # https://pymc-devs.github.io/pymc/modelchecking.html
    
    assert(0.0 <= burning <= 1.0), f"burning must be between 0 and 1!"

    ndim = len(self.bounds)

    if hdf5_file:
        f = h5py.File(hdf5_file, 'r')
        index = f['INDEX'].value[0]
        conver_steps = f['CONVER_STEPS'].value[0]
        converge_time = f['ITER_LAST'].value[0]
        chains = f['CHAINS'].value[:,:,:index+1,:]
        f.close()

        burning = int(burning*index)
        last_it = int(converge_time / conver_steps)
        chains = chains[0,:,burning:last_it,:]
    
    print("--> Performing Geweke test")

    # Convergence test over temperature 0
    #burned_chains = chains[:,burning_idx:index+1,:]  # sampler.chain
    current_length = chains.shape[1]

    # Make two subsamples at 10% and 50%
    subset_first_10 = chains[:,:int(current_length/10),:]
    subset_second_50 = chains[:,int(current_length/2):,:]

    # Divide the second half of the burned chains in 20 chunks
    chunks_idx = [int(i) for i in 
            np.linspace(0,subset_second_50.shape[1],21)][1:-1]
    subsets_20 = np.split(subset_second_50, 
                        indices_or_sections=chunks_idx, axis=1)

    # Make a z-test for the 20 chunks of the second half for
    # all the dimensions
    Z = []
    for dimension in range(ndim):
        ztas = []
        for sub20 in subsets_20:
            z = _z_score(subset_first_10[:,:,dimension], 
                                        sub20[:,:,dimension])
            ztas.append(z)
        Z.append(ztas)
    
    return Z


#@staticmethod
def _z_score(theta_a, theta_b):

    z = (np.mean(theta_a) - np.mean(theta_b)) / np.sqrt(np.var(theta_a) 
        + np.var(theta_b))
    
    return z



def logp(x):
    return 0.0


def intervals(frontiers, flat_params):
    """A function to probe wheter values are inside the stablished boundaries
    
    Arguments:
        frontiers {list} -- A list of the boundaries with [min, max] values
        flat_params {list} -- A list of flat parameters
    
    Returns:
        List -- A list with boolean values. True's are for those parameters
            inside the boundaries and False's for those outside.
    """

    TF = []
    for i in range(len(flat_params)):
        if frontiers[i][0]<= flat_params[i] <= frontiers[i][1]:
            TF.append(True)
        else:
            TF.append(False)
    return TF


def writefile(archivo , escritura , texto, align):
    
    txs = tuple(texto.split())
    
    with open(archivo, escritura) as myfile:
        myfile.write(align % txs)
        myfile.close()

    return