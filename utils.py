import numpy as np
import ttvfast
import sys
import h5py
import pickle



__all__ = ['run_TTVFast', 'calculate_epochs', 'log_likelihood_func',
            'initial_walkers', 'mcmc_summary', 'extract_best_solutions', 
            'get_mcmc_results', 'geweke', 'load_pickle', ]


__doc__ = f"Miscelaneous functions. Available are: {__all__}"


# Helpful variables
Returns = {"OPT1": np.inf, "OPT2": 1e50, "OPT3": -1.0,
            "MCMC1": -np.inf, "MCMC2": -1e50, "MCMC3": 1.0} 

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
            mass= planet[0]*Mearth_to_Msun, 
            period = planet[1], 
            eccentricity = planet[2],
            inclination = planet[3],
            argument = planet[4], 
            mean_anomaly = planet[5], 
            longnode = planet[6] 
            )
            ) 

    signal = ttvfast.ttvfast(
            planets_list, stellar_mass=mstar, time=Tin, dt=min_period/30., 
            total=Ftime, rv_times=None, input_flag=1)   
    
    SP = signal['positions']

    try:
        lastidx = SP[2].index(-2.0) # -2.0 is indicative of empty data
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
    try:        
        # Save the transit epochs of every planet and convert them 
        # to julian days. Time T0JD is in julian days. 
        # This is the reference epoch for the whole simulations
        T0 = self.T0JD 
        for k, v in self.planets_IDs.items(): 
            EPOCHS[k] = {item[1]:round(T0+item[2],6) for item in list(zip(*SP)) 
                        if item[0]==v and item[3]<= self.rstarAU}
    except:
        print('Warning: Invalid proposal')
        
    return EPOCHS    


def log_likelihood_func(flat_params, self, flag="OPT"):
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
    # (count the number of parameters outside the boundaries)
    inside_bounds = intervals(self.bounds, flat_params) 
    if False in inside_bounds:
        falses = len(inside_bounds) - sum(inside_bounds)
        return Returns[flag+"2"]  * falses  
    else:
        pass

    # Reconstruct flat_params adding the constant values
    flat_params = list(flat_params)
    for k, v in sorted(self.constant_params.items(), key=lambda j:j[0]):
        flat_params.insert(k, v)

    # Get 'positions' from signal in TTVFast
    signal_pos = run_TTVFast(
            flat_params,  mstar=self.mstar, NPLA=self.NPLA, 
            Tin=0., Ftime= self.time_span )

    # Compute simulated ephemerids (epochs: transits)
    EPOCHS = calculate_epochs(signal_pos, self)

    """
    # Compute chi^2
    # TODO: return individual chi2 for each planet. necessary??
    try:
        chi2 = 0.0
        for plnt_id, ttvs_obs in self.TTVs.items():
            ttvs_sim = EPOCHS[plnt_id]

            for epoch, times in ttvs_obs.items():
                sig_obs = (times[1] + times[2])/2.
                chi2 = chi2 + ((times[0] - ttvs_sim[epoch] )/sig_obs)**2
            
        return  Returns[flag+"3"] * chi2

    except:
        #print("EPOCHS", EPOCHS)
        return  1e10 * Returns[flag+"3"] 
    """

    # Compute the log-likelihood
    # TODO: return individual chi2 for each planet. necessary??
    loglike = 0.0
    try:
        for plnt_id, ttvs_obs in self.TTVs.items():

            chi2 = 0.0            
            ttvs_sim = EPOCHS[plnt_id]
            
            for epoch, times in ttvs_obs.items():
                sigma = self.sigma_obs[plnt_id][epoch] 
                chi2 += ((times[0] - ttvs_sim[epoch] ) / sigma)**2

            loglike += - 0.5*chi2 - self.second_term_logL[plnt_id]
        
        return Returns[flag+"3"] * loglike

    except:

        return  Returns[flag+"2"] 

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
        return _func_gp(self, distribution, ntemps=ntemps, nwalkers=nwalkers,
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
    POP0 = np.array(POP0).T.reshape(ntemps, nwalkers, len(self.bounds)) 

    return POP0



def _func_gp(self, distribution, ntemps=None, nwalkers=None, 
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
    # FIXME: There is a bug when opt_data is given from file. It must be
    # converted to list: np.genfromtxt('syn52.opt').tolist()
    [opt_data.remove(res) for res in opt_data if res[0]>=1e+50]
    opt_data = sorted(opt_data, key=lambda x: x[0])
    # FIXME: There is a bug here when threshold = 0. Then cut = -1
    # and opt_data remains the same.
    cut = int(len(opt_data)*threshold) - 1 # index of maximum chi2
    opt_data = opt_data[:cut]

    # Nuevo
    params = np.array([x[1:] for x in opt_data])#.T
    n_sols = len(params)
    indexes = list(self.constant_params.keys())
    POP0 = []
    for _ in range(ntemps*nwalkers):

        current_index = np.random.choice(range(n_sols))
        current_solution =  params[current_index].tolist()
        # Delete constant params
        for index in sorted(indexes, reverse=True):
            del current_solution[index]

        perturbed_solution = []
        for par_idx, param in enumerate(current_solution):
            # Take random numbers btw 5% from the solutions and the boundaries
            linf = (param - self.bounds[par_idx][0]) * np.random.uniform(0.0, 0.05)
            lsup = (self.bounds[par_idx][1] - param) * np.random.uniform(0.0, 0.05)
            delta = np.random.uniform(param-linf, param+lsup)

            perturbed_solution.append(delta)
        POP0.append(perturbed_solution)
    POP0 = np.array(POP0).reshape(ntemps, nwalkers, len(self.bounds))
    # Nuevo

    """
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
    """
    return POP0



def mcmc_summary(hdf5_file, burning=0.0):
    """Prints a summary of the mcmc and returns the posteriors with 1-sigma
    uncertainties corresponding to the median and 16th and 84th quantiles.
    
    Arguments:
        hdf5_file {[type]} -- The hdf5 file name
    
    Keyword Arguments:
        burning {float} -- A number between 0 and 1 corresponding to the 
        fraction of initial chains to be burned (default: {0.0})
    
    Returns:
        dict -- A dictionary with each parameter and the resulting median, 
        lower and upper errors respectively. 
        For example: {'mass1': [5.5, 1.0, 1.3]} (mass1=5.5_{-1.0}^{+1.3}), where
        5.5 is the median value of the distributions after the burning
        1.0 is the lower error
        1.3 is the upper error
    """

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
    ni = f["ITMAX"].value[0]
    cs = f["CONVER_STEPS"].value[0]
    
    maxc2 = f["BESTCHI2"].value[:index+1]
    bs = f["BESTSOLS"].value[:index+1]

    ref_epoch = f["REF_EPOCH"].value[0]
    
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
    print("   Number of iterations: ", ni)
    print("   Thining: ", cs)
    print("--------------------------")
    print("      RESULTS             ")
    print("-->Results in File:  ", hdf5_file)
    print("-->Reference epoch of the solutions: ", ref_epoch, " [JD]")
    print("-->Best solution in MCMC")
    print("   Best chi2 solution: ", round(best_chi2,5))
    for i in range(npla):
        print("   " + "   ".join( str(round(k,4)) for k in np.array_split(best_sol, npla)[i]) )
    print("--------------------------")    
    print("-->MCMC medians and 1-sigma errors")

    posteriors = {}
    #lower_err = {}
    #upper_err = {}
    for i, name in enumerate(list(colnames.split())):
        parameter = chains[:,:,i].flatten()

        low, med, up = np.percentile(parameter, [16,50,84])

        posteriors[f'{name}'], posteriors[f'{name}_e'], posteriors[f'{name}_E'] =  [med],[med-low],[up-med]
        #lower_err[f'{name}_e'] = [med-low]
        #upper_err[f'{name}_E'] = [up-med]

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
    
    return posteriors



def extract_best_solutions(hdf5_filename):
    """Run over the chains of the mcmc and extract the best solution at each 
        iteration.
    
    Arguments:
        hdf5_filename {hdf5 file} -- A file with the mcmc results.
    """

    f = h5py.File(hdf5_filename, 'r')
    mstar = f['MSTAR'][()][0]
    rstar = f['RSTAR'][()][0]
    NPLA = f['NPLA'][()][0]
    index = f['INDEX'][()][0]
    best= f['BESTSOLS'][()]
    log1_chi2 = f['BESTCHI2'][()] 
    f.close()

    # Sort solutions by chi2: from better to worst
    tupla = zip(log1_chi2[:index+1], best[:index+1])
    tupla_reducida = list(dict(tupla).items())
    sorted_by_chi2 = sorted(tupla_reducida, key=lambda tup: tup[0])[::-1] 

    best_file = '{}.best'.format(hdf5_filename.split('.')[0])
    head = "#Mstar[Msun]      Rstar[Rsun]     Nplanets"
    writefile(best_file, 'w', head, '%-10s '*3 +'\n')
    head = "#{}            {}           {}\n".format(mstar, rstar, NPLA)
    writefile(best_file, 'a', head, '%-10s '*3 +'\n')

    
    head = "#-Chi2  " + " ".join([
        "m{0}[Mearth]  Per{0}[d]  ecc{0}  inc{0}[deg]  arg{0}[deg]  M{0}[deg]\
        Ome{0}[deg] ".format(i+1) for i in range(NPLA)]) + '\n'

    writefile(best_file, 'a', head, '%-16s'+' %-11s'*(
                                            len(head.split())-1) + '\n')
    
    for _, s in enumerate(sorted_by_chi2):
        texto =  ' ' + str(round(s[0],5))+ ' ' + \
                    " ".join(str(round(i,5)) for i in s[1]) 
        writefile(best_file, 'a', texto,  '%-16s' + \
                    ' %-11s'*(len(texto.split())-1) + '\n')
    print(f'--> Best solutions from the {hdf5_filename} will be written at: {best_file}')

    return



def get_mcmc_results(hdf5_file):
    """ Extract the mcmc results from the hdf file. Returns a dictionary."""      
    
    f = h5py.File(hdf5_file, 'r')
    output = {}
    for k in f.keys():
        output[k] = f.get(k).value

    return output



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



def load_pickle(pickle_file):
    """[summary]
    
    Arguments:
        pickle_file {str} -- File name of planetary system object
    
    Returns:
        obj -- returns a planetary system object
    """
    with open(f'{pickle_file}', 'rb') as input:
        pickle_object = pickle.load(input)
    return pickle_object



def _z_score(theta_a, theta_b):

    z = (np.mean(theta_a) - np.mean(theta_b)) / np.sqrt(np.var(theta_a) 
        + np.var(theta_b))
    
    return z



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
        ##if isinstance(frontiers[i][0], list) == True:  # Omega
        ##    #for ln in frontiers[i]:
        # #   if (frontiers[i][0][0]<= flat_params[i] <= frontiers[i][0][1]) or  \
        ##    (frontiers[i][1][0]<= flat_params[i] <= frontiers[i][1][1]):  # Omega
        # #           TF.append(True)  # Omega
        ##    else:  # Omega
         #       TF.append(False)              # Omega
        #else:
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