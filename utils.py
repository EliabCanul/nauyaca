import numpy as np
import ttvfast
import sys
import h5py
import pickle

__all__ = ['run_TTVFast', 'calculate_epochs', 'log_likelihood_func',
            'initial_walkers', 'mcmc_summary', 'extract_best_solutions', 
            'get_mcmc_results', 'geweke', 'gelman_rubin', 'load_pickle', 
            'cube_to_physical', 'cube_to_physical_nomap', '_remove_constants']


__doc__ = f"Miscelaneous functions. Available are: {__all__}"

col_names = ["mass", "period", "ecc", "inclination", "argument", "mean_anomaly",
             "ascending_node"]

units = ["[M_earth]", "[d]", "", "[deg]", "[deg]", "[deg]", "[deg]"]

# Helpful variables
Returns = {"OPT1": np.inf, "OPT2": 1e100, "OPT3": -1.0,
            "MCMC1": -np.inf, "MCMC2": -1e100, "MCMC3": 1.0} 

Mearth_to_Msun = 3.0034893488507934e-06

f = lambda x: x-360 if x>360 else (360+x if x<0 else x) #lambda x: x-360 if x>360  else x

def run_TTVFast(flat_params, mstar=None, NPLA=None, Tin=0., Ftime=None, dt=None):
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
    # TODO: This can be changed to np.array(np.split(np.array(flat_params),2))
    # Test velocity..
    iters = [iter(flat_params)] * 7
    planets = list(zip(*iters))

    # TODO: Verify that planets contains exactly 7 parameters per planet

    # Define the timestep
    if dt == None:
        min_period = min([ p[1] for p in planets ])
        dt = min_period/30.

    # Iteratively adds planet's parameters to TTVFast
    planets_list = []
    for _, planet in enumerate(planets):
        # Be careful with the order!: m, per, e, inc, omega, M, Omega
        # See angle definitions: 
        # https://rebound.readthedocs.io/en/latest/python_api.html
        # TODO: refer to Nauyaca documentation (when exist)
        
        ##w = (planet[4] + planet[5])/2.
        ##M = w - planet[5]
        #print(planet[4] , planet[5], '|' ,w,M)
        planets_list.append(
        ttvfast.models.Planet( 
            mass= planet[0]*Mearth_to_Msun, 
            period = planet[1], 
            eccentricity = planet[2],
            inclination = planet[3],
            argument =  planet[4],       
            mean_anomaly =  planet[5],  
            longnode = planet[6] 
            )
            ) 

    signal = ttvfast.ttvfast(
            planets_list, stellar_mass=mstar, time=Tin, dt=dt, 
            total=Ftime, rv_times=None, input_flag=1)   
    
    SP = signal['positions']

    # TODO: checar si se pueden aplicar mascaras de numpy a este codigo
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
        ##T0 = self.T0JD 
        for k, v in self.planets_IDs.items(): 
            EPOCHS[k] = {item[1]:round(self.T0JD + item[2],8) for item in list(zip(*SP)) 
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
    inside_bounds = intervals(self.hypercube, flat_params) 
    if False in inside_bounds:
        return Returns[flag+'1'] 
    else:
        pass
    

    flat_params = cube_to_physical(self, flat_params)

    # Get 'positions' from signal in TTVFast
    signal_position = run_TTVFast(flat_params,  
                                mstar=self.mstar, NPLA=self.NPLA, 
                                Tin=0., Ftime=self.time_span, dt=self.dt)

    # Compute simulated ephemerids (epochs: transits)
    EPOCHS = calculate_epochs(signal_position, self)

    """
    if flag == 'OPT':
        # Compute chi^2
        # TODO: return individual chi2 for each planet. necessary??
        try:
            chi2 = 0.0
            for plnt_id, ttvs_obs in self.TTVs.items():
                ttvs_sim = EPOCHS[plnt_id]

                for epoch, times in ttvs_obs.items():
                    sig_obs = (times[1] + times[2])/2.
                    chi2 = chi2 + ((times[0] - ttvs_sim[epoch] )/sig_obs)**2
                
            return  Returns[flag+"3"] * np.log10(chi2)

        except:
            #print("EPOCHS", EPOCHS)
            return  Returns[flag+"2"] 
    """

    """
    # Compute the log-likelihood
    # TODO: return individual chi2 for each planet. necessary??
    # TODO: Surely it is possible to make a better function to calculate loglike
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
        # Catch situations where there is not enough simulated transits
        # to compare with observations. Return current loglike + chi2 + a constant.
        return Returns[flag+"3"] * (loglike -chi2) + Returns[flag+"2"]
    """
    loglike = 0.0

    for plnt_id, ttvs_obs in self.TTVs.items():

        chi2 = 0.0            
        ttvs_sim = EPOCHS[plnt_id]

        for epoch, times in ttvs_obs.items():
            sigma = self.sigma_obs[plnt_id][epoch] 

            try:
                chi2 += ((times[0] - ttvs_sim[epoch] ) / sigma)**2 
                last_valid_epoch = epoch
                
            except:
                #return Returns[flag+'2'] 
                """
                ###
                fracc = len(ttvs_sim)/float(len(ttvs_obs))

                if fracc > 0.9:                     
                    
                    print(last_valid_epoch, epoch,  fracc)

                    estimated_period = ttvs_sim[last_valid_epoch] - ttvs_sim[last_valid_epoch-1]
                    sim_time = missing_transits*estimated_period + ttvs_sim[last_valid_epoch]

                    #print("  sim_time:", sim_time)
                    chi2 += ((times[0] - sim_time ) / sigma)**2   # ( /sigma)**2

                    #chi2 += (1./sigma)**2                
                    missing_transits+=1
                else:
                    chi2 += - Returns[flag+"2"]
                    break
                """
                #chi2 +=  ( times[0]/sigma)**2   
                try:
                    #chi2 += np.log(1. + (times[0] - ttvs_sim[last_valid_epoch])**2/(2.*sigma**2) )
                    chi2 +=  ( (times[0]- ttvs_sim[last_valid_epoch]) /sigma)**2   
                except:
                    return Returns[flag+"2"]    

        
        currentL = - 0.5*chi2 - self.second_term_logL[plnt_id] #self.second_term_logL[plnt_id] -(3./2)*chi2
        
        loglike += currentL
        
    return  Returns[flag+"3"]*loglike


def cube_to_physical(self, x):
    
    x =  self.bi + np.array(x)*(self.bf - self.bi)
    # Reconstruct flat_params adding the constant values
    x = list(x)  
    for k, v in self.constant_params.items(): 
        x.insert(k, v)
    x = np.array(np.split(np.array(x), self.NPLA))

    w = (x[:,4] + x[:,5])/2.
    M = w - x[:,5]
    x[:,4] = list(map(f,w))
    x[:,5] = list(map(f,M)) 
    x = x.flatten()

    return x

def cube_to_physical_nomap(self, x):
    
    x =  self.bi + np.array(x)*(self.bf - self.bi)
    # Reconstruct flat_params adding the constant values
    x = list(x)  
    for k, v in self.constant_params.items(): 
        x.insert(k, v)
    x = np.array(np.split(np.array(x), self.NPLA))

    w = (x[:,4] + x[:,5])/2.
    M = w - x[:,5]
    x[:,4] = list(map(f,w+M))
    x[:,5] = list(map(f,M)) 
    x = x.flatten()

    return x

def _remove_constants(self, x):
    indexes_remove = list(self.constant_params.keys())
    x=list(x)
    for index in sorted(indexes_remove, reverse=True):
        del x[index]
    return x


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

    if nwalkers < 2*self.NPLA*7:
        sys.exit("Number of walkers must be >= 2*ndim, i.e., \n \
            nwalkers >= {}.\n Stopped simulation!".format(2*self.NPLA*7))

    if distribution.lower() == 'uniform':
        return func_uniform(self, ntemps=ntemps, nwalkers=nwalkers)

    if distribution.lower() in ("gaussian", "picked") and opt_data is not None:
        assert(0.0 < threshold <= 1.0), f"threshold must be between 0 and 1!"
        return _func_gp(self, distribution, ntemps=ntemps, nwalkers=nwalkers,
                    opt_data=opt_data, threshold=threshold)

    if distribution.lower() == 'smart':
        return _func_smart(self, distribution, ntemps=ntemps, nwalkers=nwalkers,
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
    for i in self.hypercube:
        linf = i[0]
        lsup = i[1]

        # Create uniform random walkers between boundaries
        RDM = np.random.uniform(linf , lsup, ntemps*nwalkers)
        POP0.append(RDM)
    POP0 = np.array(POP0).T.reshape(ntemps, nwalkers, len(self.hypercube)) 

    return POP0



def _func_gp(self, distribution, ntemps=None, nwalkers=None, 
            opt_data=None, threshold=1.0):

    print("\n--> Selected distribution: {}".format(distribution))

    # Clean and sort results. Get just data inside threshold.
    # FIXME: There is a bug when opt_data is given from file. It must be
    # converted to list: np.genfromtxt('syn52.opt').tolist()
    [opt_data.remove(res) for res in opt_data if res[0]>=1e+50]
    opt_data = sorted(opt_data, key=lambda x: x[0])
    original_len = len(opt_data)

    cut = int(len(opt_data)*threshold) # index of maximum chi2
    opt_data = opt_data[:cut]

    params = np.array([x[1:] for x in opt_data])
    print(f"    {len(opt_data)} of {original_len} solutions taken")

    POP0 = []

    if distribution.lower() == 'picked':
        n_sols = len(params)
        indexes = list(self.constant_params.keys())

        for _ in range(ntemps*nwalkers):
            current_index = np.random.choice(range(n_sols))
            current_solution =  params[current_index].tolist()
            
            # Delete constant params
            #for index in sorted(indexes, reverse=True):
            #    del current_solution[index]

            perturbed_solution = []
            rdmu_b = np.random.uniform(0.0, 0.1)
            for par_idx, param in enumerate(current_solution):
                # Take random numbers btw 3% from the solutions and the boundaries
                #linf = (param - self.bounds[par_idx][0]) * np.random.uniform(0.0, 0.03)
                #lsup = (self.bounds[par_idx][1] - param) * np.random.uniform(0.0, 0.03)
                linf = (param - self.hypercube[par_idx][0]) * rdmu_b
                lsup = (self.hypercube[par_idx][1] - param) * rdmu_b                
                delta = np.random.uniform(param-linf, param+lsup)

                perturbed_solution.append(delta)
            POP0.append(perturbed_solution)
        POP0 = np.array(POP0).reshape(ntemps, nwalkers, len(self.bounds))
    
    if distribution.lower() == 'gaussian':
        params = params.T
        i = 0
        for par_idx, param in enumerate(params):
            # Initialize walkers avoiding the constant paremeters
            #if par_idx not in list(self.constant_params.keys()):
            poptmp = [] 
            while len(poptmp) < ntemps*nwalkers:
                # Calculate parameters for the gaussian distribution
                mu = np.mean(param)
                sig = np.std(param)

                if sig == 0.0:
                    rdm = mu
                else:
                    rdm = np.random.normal(loc=mu,scale=sig)

                #if self.bounds[i][0] <= rdm and rdm <= self.bounds[i][1]:
                if self.hypercube[i][0] <= rdm and rdm <= self.hypercube[i][1]:
                    poptmp.append(rdm)

            POP0.append(poptmp)
            i += 1
        POP0 = np.array(POP0).T.reshape(ntemps, nwalkers, len(self.bounds))

    return POP0

# TODO: Cambiar a Ladder
def _func_smart(self, distribution, ntemps=None, nwalkers=None,
                    opt_data=None, threshold=None):
    print("\n--> Selected distribution: {}".format(distribution))      
    
    [opt_data.remove(res) for res in opt_data if res[0]>=1e+50]
    acomodar = sorted(opt_data, key=lambda x: x[0])
    original_len = len(acomodar)

    cut = int(len(acomodar)*threshold) # index of maximum chi2
    acomodar = acomodar[:cut]

    f = lambda x: x[1:]
    parameters = list(map(f, acomodar ))
    print(f"    {len(acomodar)} of {original_len} solutions taken")

    #parameters_sep = list(_chunks(parameters, ntemps)) 
    ##indexes = list(self.constant_params.keys())

    POP0 = []
    #for par_sep in parameters_sep:  # Iterates over chunks (temperatures)
    for pt in range(ntemps):  # Iterates over chunks (temperatures)
        #
        parameters_sep = list(_chunks(parameters, ntemps-pt)) 
        par_sep = parameters_sep[0]
        #
        
        n_sols=len(par_sep)
        #print('n_sols: ', n_sols )
        par_sep_T = list(np.array(par_sep).T)

        # Delete constant params
        #for index in sorted(indexes, reverse=True):
        #    del par_sep_T[index]
        
        par_sep_2 = np.array(par_sep_T).T
        #pprint(par_sep_2)
        
        # choose randomly a index in the chunk
        current_index = np.random.choice(range(n_sols), nwalkers ) 
        #print('current_index: ', current_index)    
        
        for i in current_index:
            current_solution = par_sep_2[i]
            
            perturbed_solution = []
            rdmu_b = np.random.uniform(0.0, 0.1)
            for par_idx, param in enumerate(current_solution):
                # Take random numbers btw 3% from the solutions and the boundaries
                #linf = (param - self.bounds[par_idx][0]) * rdmu_b
                #lsup = (self.bounds[par_idx][1] - param) * rdmu_b
                linf = (param - self.hypercube[par_idx][0]) * rdmu_b
                lsup = (self.hypercube[par_idx][1] - param) * rdmu_b
                delta = np.random.uniform(param-linf, param+lsup)

                perturbed_solution.append(delta)   
        
            POP0.append(perturbed_solution)
        #print('len POP:', len(POP0))    
        
        #print('-------')    

    POP0 = np.array(POP0).reshape(ntemps, nwalkers, len(self.hypercube))  
    return POP0




def _chunks(l, n):
    """Yield n number of sequential chunks from l."""
    d, r = divmod(len(l), n)
    for i in range(n):
        si = (d+1)*(i if i < r else r) + d*(0 if i < r else i - r)
        yield l[si:si+(d+1 if i < r else d)]


def mcmc_summary(self, hdf5_file, burning=0.0, fthinning=1, verbose=True):
    """Prints a summary of the mcmc and returns the posteriors with 1-sigma
    uncertainties corresponding to the median and 16th and 84th quantiles.
    
    Arguments:
        hdf5_file {[type]} -- The hdf5 file name
    
    Keyword Arguments:
        burning {float} -- A number between 0 and 1 corresponding to the 
        fraction of initial chains to be burned (default: {0.0})
        fthinning -- Thinning factor of the chain. Default 1 takes all the
        the saved chains. Real thinning factor is fthining*CONVER_STEPS.
        verbose -- Print a summary
    
    Returns:
        dict -- A dictionary with the parameter name and a its respective
        flat array along the parameter dimension
        For example: {'mass1': [5.5, 1.0, 1.3]} (mass1=5.5_{-1.0}^{+1.3}), where
        5.5 is the median value of the distributions after the burning
        1.0 is the lower error
        1.3 is the upper error
    """

    assert(0.0 <= burning <= 1.0), f"burning must be between 0 and 1!"
    assert( isinstance(fthinning, int) ), "fthinning should be int"

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
    
    burning_frac = burning

    burning = int(burning*index)
    # shape for chains is: (temps,walkers,steps,dim)
    chains = f['CHAINS'].value[0,:,burning:index+1:fthinning,:] #Just for temperature 0

    f.close()
    
    best = zip(maxc2, bs)

    # Reverse the list because we sorted by -chi**2
    sort_res = sorted(list(best), key=lambda j: j[0], reverse=True)
    best_chi2, best_sol = sort_res[0][0], sort_res[0][1]

    #chains = chains[:,burning:index+1,:]
    if verbose:
        print("-->Planetary System: ", syst_name)
        print("   Stellar mass: ", ms)
        print("   Stellar radius: ", rs)
        print("   Number of planets: ", npla)
        print("--------------------------")
        print("-->MCMC parameters")
        print("   Ntemps: ", nt)
        print("   Nwalkers per temperature: ", nw)
        print("   Number of iterations: ", ni)
        print("   Thining: ", cs*fthinning)
        print("   Burning: ", burning_frac)
        print("   Chain shape: ", chains.shape)
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

    # Convert normalized chains to physical values
    chains = np.array([[cube_to_physical(self, x) for x in chains[w,:,:]] for w in range(nw) ])
    chains = np.array([[_remove_constants(self, x) for x in chains[w,:,:]] for w in range(nw) ])
    #

    for i, name in enumerate(list(colnames.split())):
        parameter = chains[:,:,i].flatten()

        low, med, up = np.percentile(parameter, [16,50,84])

        #posteriors[f'{name}'], posteriors[f'{name}_e'], posteriors[f'{name}_E'] =  [med],[med-low],[up-med]
        posteriors[f'{name}'] = parameter

        if verbose:
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
    names = f['COL_NAMES'][()]
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

    
    #head = "#-Chi2  " + " ".join([
    #    "m{0}[Mearth]  Per{0}[d]  ecc{0}  inc{0}[deg]  arg{0}[deg]  M{0}[deg]\
    #    Ome{0}[deg] ".format(i+1) for i in range(NPLA)]) + '\n'
    head = "#-chi2   " + names #"  ".join([na for na in names.split()])

    writefile(best_file, 'a', head, '%-16s'+' %-11s'*(
                                            len(head.split())-1) + '\n')
    
    for _, s in enumerate(sorted_by_chi2):
        texto =  ' ' + str(s[0])+ ' ' + \
                    " ".join(str(i) for i in s[1]) 
        writefile(best_file, 'a', texto,  '%-30s' + \
                    ' %-11s'*(len(texto.split())-1) + '\n')
    print(f'--> Best solutions from the {hdf5_filename} will be written at: {best_file}')

    return sorted_by_chi2



def get_mcmc_results(hdf5_file, keywords=None):
    """ Extract the mcmc results from the hdf5 file. Returns a dictionary."""      
    
    f = h5py.File(hdf5_file, 'r')
    if keywords is not None:
        for keys in keywords:
            assert(keys in f.keys()), f"Keyword -{keys}- does not exists in File."+\
            f" Available keywords are : {list(f.keys())}"
    else:
        keywords = f.keys()    

    output = {}
    for k in keywords:
        output[k] = f[k].value #f.get(k).value

    return output


def gelman_rubin(chains, nchunks_gr=10, thinning=10, names=None):

    assert(len(chains.shape) == 3), "Shape for chains should be: (walkers,steps,dim)"+\
        f" instead of {chains.shape}"
    if names is None:
        # Create a generic list of names
        names = [f"dim{d}" for d in list(range(chains.shape[-1]))] 
    
    # Select the steps to perform GR statistic 
    steps = [ int(it) for it in np.linspace(0,chains.shape[1],nchunks_gr+1)[:-1] ]

    GR_statistic = {}
    print("--> Performing Gelman-Rubin test")
    for param, param_name in enumerate(names):

        GRmean = []
        for it in steps:

            # Selected chains
            X = chains[::thinning, it:, param]

            J = X.shape[0] # number of chains (walkers)
            L = X.shape[1] # number of steps

            # chain mean
            xmean_j = list(map(np.mean, X))

            # grand mean
            xmean_dot = np.mean(xmean_j) 

            # Between chain variance
            between = lambda x: (x -xmean_dot)**2

            B = (L/(J-1)) * sum(list(map(between, xmean_j)))

            # within chain variance
            within = lambda x: (x - xmean_j)**2

            GR = []
            for j in range(J): # Over chains (walkers)

                s2j = (1./(L-1)) * sum(list(map(within, X[j])))

                W = np.mean(s2j)

                Var = ((L-1)/L)*W + (1./L)*B 

                R = np.sqrt(Var / W)

                GR.append(R)

            GRmean.append(np.mean(GR))  # mean GR over chains

        GR_statistic[param_name] = GRmean
    
    return GR_statistic


def geweke(self, hdf5_file=None, chains=None, names=None, burning=0):
    # Geweke criterion
    # https://rlhick.people.wm.edu/stories/bayesian_5.html
    # https://pymc-devs.github.io/pymc/modelchecking.html
    
    assert(0.0 <= burning <= 1.0), f"burning must be between 0 and 1!"

    ##ndim = len(self.bounds)

    if hdf5_file:
        f = h5py.File(hdf5_file, 'r')
        index = f['INDEX'].value[0]
        burning = int(burning*index)
        #conver_steps = f['CONVER_STEPS'].value[0]
        #converge_time = f['ITER_LAST'].value[0]
        chains = f['CHAINS'].value[0,:,burning:index+1,:]
        names = list(f['COL_NAMES'].value[:].split())
        f.close()

        
        #last_it = int(converge_time / conver_steps)
        #chains = chains[0,:,burning:last_it,:]
        #chains = chains[0,:,burning:index+1,:]
    
    if names is None:

        names = list(range(chains.shape[-1]))


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
    Z = {}
    for dimension in range(self.ndim):
        ztas = []
        for sub20 in subsets_20:
            z = _z_score(subset_first_10[:,:,dimension], 
                                        sub20[:,:,dimension])
            ztas.append(z)
        Z[names[dimension]] = ztas
    
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