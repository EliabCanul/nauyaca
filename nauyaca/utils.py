import numpy as np
import ttvfast
import sys
import h5py
from .constants import Mearth_to_Msun, col_names


__all__ = ['run_TTVFast', 'calculate_ephemeris', 'log_likelihood_func',
            'init_walkers', 'mcmc_summary', 'extract_best_solutions', 
            'get_mcmc_results', 'geweke', 'gelman_rubin',  
            'cube_to_physical', '_ephemeris', '_remove_constants']

__doc__ = f"Miscelaneous functions to support the main modules. Available are: {__all__}"


def run_TTVFast(flat_params, mstar, init_time=0., final_time=None, dt=None):
    """A function to communicate with the wrapper around TTVFast

    Parameters
    ----------
    flat_params : array or list
        A flat array containing the seven planet parameters of the first planet
        concatenated with the seven parameters of the next planet and so on. 
        Thus, it should be of length 7 x number of planets.
        The order per planet must be: 
        mass [Mearth], period [days], eccentricity, inclination [deg], argument
        of periastron [deg], mean anomaly [deg] and ascending node [deg].
    mstar : float
        The stellar mass [Msun].
    init_time : float
        Initial time of the simulations [days], by default 0.
    final_time : float
        Final time of the simulations [days], by default None.
    dt : float
        Timestep of the simulations [days], by default None.

    Returns
    -------
    array
        An array with transit numbers and transit epochs for all 
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
    for planet in planets:
        # Be careful with the order!: m, per, e, inc, omega, M, Omega
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
            planets_list, stellar_mass=mstar, time=init_time, total=final_time,
            dt=dt, rv_times=None, input_flag=1)   
    
    SP = np.array(signal['positions'])

    # -2.0 is indicative of empty data
    mask = [SP[2] != -2]
    
    SP = np.array([ i[mask] for i in SP])
    
    return SP


def _ephemeris(PSystem, SP):
    """Calculates the simulated ephemeris per planet from the TTVFast result

    Parameters
    ----------
    PSystem : 
        The Planetary System object.
    SP : array
        An array with transit numbers and transit epochs for all 
        planets labeled in the entrance order from flat_params. 
        From TTVFast doc:
        [PLANET,  EPOCH,  TIME (DAYS),  RSKY (AU),  VSKY (AU/DAY)]
        SP[0] = planet index, SP[1] = transit number, SP[2] = central time, 
        SP[3] = Rsky, SP[4] = Vsky

    Returns
    -------
    dict
        A dictionary where keys are the planet_IDs from PSystem and the values
        are the simulated times of transit
    """
    
    ephemeris = {}
    try:        
        # Save the transit epochs of every planet and convert them 
        # to julian days. Time t0 is in julian days. 
        for planet_id, planet_number in PSystem.planets_IDs.items(): 

            planet_mask = SP[0] == planet_number
            transit_mask = SP[3][planet_mask] <= PSystem.rstarAU

            transit_number = SP[1][planet_mask][transit_mask].astype(int)
            transit_time = SP[2][planet_mask][transit_mask] 
            
            ephemeris[planet_id] = dict(zip(transit_number,transit_time))

    except:
        print('Warning: Invalid proposal')
        
    return ephemeris    


def calculate_ephemeris(PSystem, flat_params):
    """A function to calculate simulated ephemeris from normalized solutions

    Parameters
    ----------
    PSystem : 
        The Planetary System object
    flat_params : array or list
        A flat array containing the planet parameters of the first planet
        concatenated with the parameters of the next planet and so on. 
        The order per planet must be: 
        mass, period, eccentricity, inclination, argument of periastron, mean 
        anomaly and ascending node.
        Unlike to flat_params in run_TTVFast(), flat_params here is in the
        normalized version (between 0 and 1) and also the constant parameters
        should not be included.

    Returns
    -------
    dict
        A dictionary where keys are the planet_IDs from PSystem and the values
        are the simulated times of transit
    """

    # Convert from hypercube data to physical
    flat_params = cube_to_physical(PSystem, flat_params)

    # Get 'positions' from signal in TTVFast
    signal_position = run_TTVFast(flat_params,  
                                PSystem.mstar, 
                                init_time=PSystem.t0, 
                                final_time=PSystem.ftime, 
                                dt=PSystem.dt)

    # Compute simulated ephemerids (epochs: transits)
    epochs = _ephemeris(PSystem, signal_position)
    
    return epochs


def _chi2(observed, sigma, simulated, individual=False):
    """A function to calculate chi square statistic from observed and simulated
    ephemeris

    Parameters
    ----------
    observed : dict
        A dictionary where keys are the epochs and values the observed mid-transit
        times
    sigma : dict
        A dictionary where keys are the epochs and values the uncertainties in
        the observed mid-transit times
    simulated : dict
        A dictionary where keys are the epochs and values the simulated 
        mid-transit times
    individual : bool, optional
        A flag to return individual chi squaere per planet, by default False

    Returns
    -------
    float (individual=False)
        The sum of the chi square of all the planets in the system

    dict (if individual=True)
        The chi square per planet in the system
    """
    
    if individual:
        ind_chi2 = {}
        
    chi2_tot = 0.0
    for planet_id, obs in observed.items():

        chi2 = 0.0

        sim = simulated[planet_id]
        sig = sigma[planet_id]  

        for epoch, times_obs in obs.items():
            try:
                chi2 += ((times_obs - sim[epoch])/sig[epoch])**2

            except:
                # Add a high constant each time a simulated transit 
                # is not detected
                chi2 += 1e+20
    
        chi2_tot += chi2
        
        if individual:
            ind_chi2[planet_id] = chi2
    
    if individual:
        return ind_chi2
    else:
        return chi2_tot


def calculate_chi2(flat_params, PSystem):
    """A function to calculate chi square from normalized solutions

    This function is used by optimizers.

    Parameters
    ----------
    flat_params : array or list
        A flat array containing the planet parameters of the first planet
        concatenated with the parameters of the next planet and so on. 
        The order per planet must be: 
        mass, period, eccentricity, inclination, argument of periastron, mean 
        anomaly and ascending node.
        Unlike to flat_params in run_TTVFast(), flat_params here is in the
        normalized version (between 0 and 1) and also the constant parameters
        should not be included.
    PSystem : 
        The Planetary System object

    Returns
    -------
    float
        The chi square of all the planets in the system
    """

    # Verify that proposal is inside the boundaries.
    if False in intervals(PSystem.hypercube, flat_params):
        return np.inf 
    else:
        pass
    
    sim_times = calculate_ephemeris(PSystem, flat_params)

    chi2 = _chi2(PSystem.transit_times, PSystem.sigma_obs, sim_times, individual=False)

    return chi2


def calculate_chi2_physical(flat_params, PSystem, individual=False, 
                            insert_constants=False, get_ephemeris=False):
    """Calculates the chi square of TTVs fitting given a set of planetary parameters.

    If required, it also returns the simulated ephemeris per planet

    This function do not verify that planetary parameters are inside the 
    boundaries in PSystem.bounds

    Parameters
    ----------
    flat_params : array or list
        A flat array containing the planet parameters of the first planet
        concatenated with the parameters of the next planet and so on. 
        The order per planet must be: 
        mass [Mearth], period [days], eccentricity, inclination [deg], argument
        of periastron [deg], mean anomaly [deg] and ascending node [deg].
    PSystem : 
        The Planetary System object
    individual : bool, optional
        A flag to return individual chi squares of the planets, by default False.
    insert_constants : bool, optional
        A flag to insert the constant parameters from PSystem.constant_params, 
        by default False. 
        If set to False, flat_params must be of length 7 x number of planets. 
        If set to True, then flat_params should not include the constant parameters.
    get_ephemeris : bool, optional
        A flag to get the simulated ephemeris resulting from the flat_params,
        by default False. If set to True, then returns a tuple with the simulated
        ephemeris per planet

    Returns
    -------
    float (individual=False)
        The sum of the chi square of all the planets in the system

    dict (if individual=True)
        The chi square per planet in the system
    
    tuple (float/dict, dict)
        If get_ephemeris=True, then return the chi square and the simulated 
        ephemeris
    """

    x = list(flat_params)  

    if insert_constants:
        for k, v in PSystem.constant_params.items(): 
            x.insert(k, v)

    if len(x) == 7 * PSystem.npla:
        pass 
    else:
        sys.exit("Length of -flat_params- mismatch number of dimensions per planet")

    # Get 'positions' from signal in TTVFast
    signal_position = run_TTVFast(x,  
                                PSystem.mstar, 
                                init_time=PSystem.t0, 
                                final_time=PSystem.ftime, 
                                dt=PSystem.dt)

    # Compute simulated ephemerids (epochs: transits)
    sim_times = _ephemeris(PSystem, signal_position)
    
    chi2 = _chi2(PSystem.transit_times, PSystem.sigma_obs, sim_times, 
                individual=individual)

    if get_ephemeris:
        return (chi2, sim_times)
    else:
        return chi2


def log_likelihood_func(flat_params, PSystem):
    """A function to calculate the Log Likelihood

    This function is used by the MCMC

    Parameters
    ----------
    flat_params : array or list
        A flat array containing the planet parameters of the first planet
        concatenated with the parameters of the next planet and so on. 
        The order per planet must be: 
        mass, period, eccentricity, inclination, argument of periastron, mean 
        anomaly and ascending node.
        Unlike to flat_params in run_TTVFast(), flat_params here is in the
        normalized version (between 0 and 1) and the constant parameters
        should not be included.
    PSystem : 
        The Planetary System object

    Returns
    -------
    float
        The log likelihood of the current solution
    """    

    # Verify that proposal is inside boundaries    
    if False in intervals(PSystem.hypercube, flat_params):
        return -np.inf
    else:
        pass
    
    sim_times = calculate_ephemeris(PSystem, flat_params)

    # Compute the log-likelihood
    chi2 = _chi2(PSystem.transit_times, PSystem.sigma_obs, sim_times, individual=False)

    loglike = - 0.5*chi2 - sum(PSystem.second_term_logL.values())
    
    return  loglike 


def cube_to_physical(PSystem, x):
    """A function to convert from normalized solutions to physicals

    This function also adds the constant values into the returned array. Thus,
    x should not include the constant parameters

    Parameters
    ----------
    PSystem : 
        The Planetary System object
    x : list
        A list with the normalized solutions (between 0 and 1)

    Returns
    -------
    array
        An array with the 7 physical planet parameters of the first planet
        concatenated with the 7 parameters of the next planet and so on. 
        It includes the constant parameters.
    """

    f = lambda x: x-360 if x>360 else (360+x if x<0 else x) 
    
    # Map from normalized to physicals
    x = np.array(PSystem.bi) + np.array(x)*(np.array(PSystem.bf) - np.array(PSystem.bi))

    # Reconstruct flat_params adding the constant values
    x = list(x)  
    for k, v in PSystem.constant_params.items(): 
        x.insert(k, v)
    x = np.array(np.split(np.array(x), PSystem.npla))

    # Invert the parameterized angles to get argument and ascending node
    w = (x[:,4] + x[:,5])/2.
    M = w - x[:,5]
    x[:,4] = list(map(f,w))
    x[:,5] = list(map(f,M)) 

    return x.flatten()


def _remove_constants(PSystem, x):
    """A help function to remove the constant parameters from x. In this case,
    x must be of length 7 x number of planets
    """

    indexes_remove = list(PSystem.constant_params.keys())
    x=list(x)
    for index in sorted(indexes_remove, reverse=True):
        del x[index]

    return np.array(x)


def init_walkers(PSystem, distribution=None, opt_data=None, ntemps=None, 
                    nwalkers=None,  fbest=1.0):
    """An useful function to easily create initial walkers.

    Parameters
    ----------
    PSystem : 
        The Planetary Sysstem object
    distribution : str, optional
        The name of the built-in distribution to create initial walkers.
        Available options are: "Uniform", "Gaussian", "Picked" and "Ladder".
        For the last three, provide the results from the optimizers through the
        opt_data kwarg, by default None.
    opt_data : array or dict, optional
        Results from the optimizers that will be used to create the population
        of walkers, by default None.
        -If dict, it have to be the dictionary comming from the optimizers with 
        keys 'chi2', 'cube', 'physical'.
        -If array, it have to be an array created from the file '*_cube.opt',
        for example, using numpy.genfromtxt()
    ntemps : int, optional
        Number of temperatures for the parallel-tempering MCMC. If p0 is not
        None, ntemps is taken from p0 shape[0].
    nwalkers : int, optional
        Number of walkers per temperature. If p0 is not None, nwalkers is taken
        from p0 shape[1].
    fbest : float, optional
        A fraction between 0 and 1 to especify the fraction of best solutions
        from opt_data (if given) to create p0, by default 1.

    Returns
    -------
    array
        The initial population of walkers. Shape must be (ntemps, nwalkers, ndim)
    """

    if nwalkers < 2*PSystem.ndim:
        raise RuntimeError(f"Number of walkers must be >= 2*ndim, i.e., " +
            f"nwalkers have to be >= {2*PSystem.ndim}.")

    if distribution.lower() == 'uniform':
        return _func_uniform(PSystem, ntemps=ntemps, nwalkers=nwalkers)

    if opt_data is not None:
        assert(0.0 < fbest <= 1.0), "fbest must be between 0 and 1!"

        if distribution.lower() in ("gaussian", "picked", "ladder"): 
            
            print("\n--> Selected distribution: {}".format(distribution))   
            
            return _func_from_opt(PSystem, distribution, ntemps=ntemps, nwalkers=nwalkers,
                        opt_data=opt_data, fbest=fbest)

        else:
            text = ("--> Argument 'distribution' does not match with any",
            " supported distribution. Available options are:",
            "\n *Uniform \n *Gaussian \n *Picked \n *Ladder" ,
            "\n For the last three, please provide results from the ",
            "optimizer routine through 'opt_data' argument.")
            raise ValueError(" ".join(text))

    else:
        print("Arguments not understood")


def _func_uniform(PSystem, ntemps=None, nwalkers=None):
    """A help function to create walkers from a uniform distribution"""

    print("\n--> Selected distribution: Uniform")
    POP0 = []
    for i in PSystem.hypercube:
        linf = i[0]
        lsup = i[1]

        # Create uniform random walkers between boundaries
        RDM = np.random.uniform(linf , lsup, ntemps*nwalkers)
        POP0.append(RDM)
    POP0 = np.array(POP0).T.reshape(ntemps, nwalkers, len(PSystem.hypercube)) 

    return POP0


def _func_from_opt(PSystem, distribution, ntemps=None, nwalkers=None, 
            opt_data=None, fbest=1.0):
    """A help function to create walkers from distributions that requiere
    information from optimizers. These are: 'Gaussian', 'Picked' and 'Ladder'.
    The main function to call this subfunction is init_walkers(). """

    # Dictionary from optimizers
    if type(opt_data) == dict:
        x = opt_data['cube']
        fun = opt_data['chi2']
        opt_data = np.column_stack((fun, x))

    # TODO: It is necessary to give the input in 'cube' format? or it can be 'phys' either?
    assert((opt_data[:,1:]>=0.).all() and (opt_data[:,1:] <=1.0).all()), "Invalid opt_data. Provide 'cube' solutions"

    # Remove the solutions where chi2 have solutions => 1e+20
    # since that value means that optimizers didn't find a solutions
    opt_data = opt_data[opt_data[:,0] <1e+20]
    opt_data = sorted(opt_data, key=lambda x: x[0])
    original_len = len(opt_data)

    cut = int(len(opt_data)*fbest) # index of maximum chi2
    opt_data = opt_data[:cut]

    POP0 = []


    if distribution.lower() == 'picked':
        params = np.array([x[1:] for x in opt_data])
        print(f"    {len(opt_data)} of {original_len} solutions taken")

        n_sols = len(params)

        for _ in range(ntemps*nwalkers):
            current_index = np.random.choice(range(n_sols))
            current_solution =  params[current_index].tolist()
            
            perturbed_solution = []
            rdmu_b = np.random.uniform(0.0, 0.1)

            for par_idx, param in enumerate(current_solution):
                linf = (param - PSystem.hypercube[par_idx][0]) * rdmu_b
                lsup = (PSystem.hypercube[par_idx][1] - param) * rdmu_b                
                delta = np.random.uniform(param-linf, param+lsup)

                perturbed_solution.append(delta)
            POP0.append(perturbed_solution)
        POP0 = np.array(POP0).reshape(ntemps, nwalkers, len(PSystem.bounds))
    

    if distribution.lower() == 'gaussian':
        params = np.array([x[1:] for x in opt_data])
        print(f"    {len(opt_data)} of {original_len} solutions taken")

        params = params.T
        i = 0
        for par_idx, param in enumerate(params):

            poptmp = [] 
            while len(poptmp) < ntemps*nwalkers:
                # Calculate parameters for the gaussian distribution
                mu = np.mean(param)
                sig = np.std(param)

                if sig == 0.0:
                    rdm = mu
                else:
                    rdm = np.random.normal(loc=mu,scale=sig)

                if PSystem.hypercube[i][0] <= rdm and rdm <= PSystem.hypercube[i][1]:
                    poptmp.append(rdm)

            POP0.append(poptmp)
            i += 1
        POP0 = np.array(POP0).T.reshape(ntemps, nwalkers, len(PSystem.bounds))


    if distribution.lower() == 'ladder':
        f = lambda x: x[1:]
        parameters = list(map(f, opt_data ))
        print(f"    {len(opt_data)} of {original_len} solutions taken")


        for pt in range(ntemps):  # Iterates over chunks (temperatures)
            #
            parameters_sep = list(_chunks(parameters, ntemps-pt)) 
            par_sep = parameters_sep[0]
            #
            
            n_sols=len(par_sep)
            par_sep_T = list(np.array(par_sep).T)
            
            par_sep_2 = np.array(par_sep_T).T
            
            # choose randomly an index in the chunk
            current_index = np.random.choice(range(n_sols), nwalkers ) 
            
            for i in current_index:
                current_solution = par_sep_2[i]
                
                perturbed_solution = []
                rdmu_b = np.random.uniform(0.0, 0.1)

                for par_idx, param in enumerate(current_solution):
                    linf = (param - PSystem.hypercube[par_idx][0]) * rdmu_b
                    lsup = (PSystem.hypercube[par_idx][1] - param) * rdmu_b
                    delta = np.random.uniform(param-linf, param+lsup)

                    perturbed_solution.append(delta)   
            
                POP0.append(perturbed_solution)
        POP0 = np.array(POP0).reshape(ntemps, nwalkers, len(PSystem.hypercube))  


    return POP0


def _chunks(l, n):
    """Yield n number of sequential chunks from l."""
    d, r = divmod(len(l), n)
    for i in range(n):
        si = (d+1)*(i if i < r else r) + d*(0 if i < r else i - r)
        yield l[si:si+(d+1 if i < r else d)]


def mcmc_summary(PSystem, hdf5_file, burnin=0.0, fthinning=1, get_posteriors=False,
                verbose=True):
    """Prints a summary of the mcmc run and returns the chains in physical
    values for individual planet parameters after the specified burnin.

    Parameters
    ----------
    PSystem : 
        The Planetary System object
    hdf5_file : str
        The name of the hdf5 file from which the summary will be extracted.
    burnin : float, optional
        A fraction between 0 and 1 to discard as burn-in at the beggining of
        the chains, by default 0.0.
    fthinning : int, optional
        A factor to thin the chains, by default 1. A fthining of 10 in a 1000
        steps chain, will return the summary for 100 steps. Recommendable for 
        longer chains.
    get_posteriors : bool, optional
        A flag to return a dictionary of posteriors, by default False. If set
        to True, then this function returns a dictionary of planet parameters
        as keys and flatten arrays along the steps as values.
    verbose : bool, optional
        A flag to allow for verbose (True) or return the posteriors in quiet
        form (False), by default True.

    Returns
    -------
    dict
        Returns a dictionary where keys are the physical planetary parameters 
        and the values are the chains of these parameters after the burnin.
    """

    # TODO: This method should be at MCMC?

    assert(0.0 <= burnin <= 1.0), f"burnin must be between 0 and 1!"
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
    cs = f["INTRA_STEPS"].value[0]
    
    maxc2 = f["BESTLOGL"].value[:index+1]
    bs = f["BESTSOLS"].value[:index+1]

    ref_epoch = f["REF_EPOCH"].value[0]
    
    burnin_frac = burnin

    burnin = int(burnin*index)
    # shape for chains is: (temps,walkers,steps,dim)
    # Only for temperature 0
    chains = f['CHAINS'].value[0,:,burnin:index+1:fthinning,:] 

    f.close()
    
    best = zip(maxc2, bs)

    # reverse=False: ascending order
    sort_res = sorted(list(best), key=lambda j: j[0], reverse=False)
    best_logl, best_sol = sort_res[-1][0], sort_res[-1][1]

    best_sol = cube_to_physical(PSystem, best_sol)

    if verbose:
        print("--> Planetary System: ", syst_name)
        print("    Stellar mass: ", ms)
        print("    Stellar radius: ", rs)
        print("    Number of planets: ", npla)
        print("--> Planets:")
        for p, n in PSystem.planets_IDs.items():
            print(f"       Planet{n+1}: {p}")
        print("    ")
        print("--------------------------")
        print("--> MCMC parameters")
        print("    Ntemps: ", nt)
        print("    Nwalkers per temperature: ", nw)
        print("    Number of iterations: ", ni)
        print("    Thining: ", cs*fthinning)
        print("    Burnin: ", burnin_frac)
        print("    Chain shape: ", chains.shape)
        print("--------------------------")
        print("      RESULTS             ")
        print("--> Results in File:  ", hdf5_file)
        print("--> Reference epoch of the solutions: ", ref_epoch, " [JD]")
        print("--> Best solution in MCMC")
        print("    Logl: ", round(best_logl,5))
        print("          "+" ".join([cn for cn in col_names]))
        for i in range(npla):
            print(f"Planet{i+1}: " + "   ".join( str(round(k,4)) for k in np.array_split(best_sol, npla)[i]) )
        print("--------------------------")    
        print("--> MCMC medians and 1-sigma errors")

    posteriors = {}

    # Convert normalized chains to physical values
    chains = np.array([[cube_to_physical(PSystem, x) for x in chains[w,:,:]] for w in range(nw) ])
    chains = np.array([[_remove_constants(PSystem, x) for x in chains[w,:,:]] for w in range(nw) ])
    #

    for i, name in enumerate(list(colnames.split())):
        parameter = chains[:,:,i].flatten()

        low, med, up = np.percentile(parameter, [16,50,84])

        if get_posteriors:
            posteriors[f'{name}'] = parameter

        if verbose:
            if i == 1: 
                # For period increase decimals
                tit = "%s ^{+%s}_{-%s} " % (round(med,5),
                                                round(up-med,5),
                                                round(med-low,5))
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
    
    if get_posteriors:
        return posteriors
    else:
        return


def extract_best_solutions(hdf5_filename, write_file=True):
    """Extract the best solutions saved in the hdf5 file.

    Parameters
    ----------
    hdf5_filename : str
        The hdf5 file from where the best solutions will be extracted.
    write_file : bool, optional
        A flag to write these solutions in a file (True), by default True.

    Returns
    -------
    list
        The result is a list of tuples where the first element is the logl and 
        the second element is an array with the corresponding solution in the
        normalized form. It is sorted from better to worse solution.
    """

    f = h5py.File(hdf5_filename, 'r')
    mstar = f['MSTAR'][()][0]
    rstar = f['RSTAR'][()][0]
    NPLA = f['NPLA'][()][0]
    index = f['INDEX'][()][0]
    best= f['BESTSOLS'][()]
    log1_chi2 = f['BESTLOGL'][()] 
    names = f['COL_NAMES'][()]
    f.close()

    # Sort solutions by chi2: from better to worst
    tupla = zip(log1_chi2[:index+1],best[:index+1])
    tupla_reducida = list(dict(tupla).items())   # Remove possible repeated data
    sorted_by_chi2 = sorted(tupla_reducida, key=lambda tup: tup[0], reverse=True)#[::-1] 

    if write_file:
        best_file = '{}.best'.format(hdf5_filename.split('.hdf5')[0])
        head = "#Mstar[Msun]      Rstar[Rsun]     Nplanets"
        writefile(best_file, 'w', head, '%-10s '*3 +'\n')
        head = "#{}            {}           {}\n".format(mstar, rstar, NPLA)
        writefile(best_file, 'a', head, '%-10s '*3 +'\n')

        
        head = "#-chi2   " + names 

        writefile(best_file, 'a', head, '%-16s'+' %-11s'*(
                                                len(head.split())-1) + '\n')
        
        for _, s in enumerate(sorted_by_chi2):
            texto =  ' ' + str(s[0])+ ' ' + \
                        " ".join(str(i) for i in s[1]) 
            writefile(best_file, 'a', texto,  '%-30s' + \
                        ' %-11s'*(len(texto.split())-1) + '\n')
        print(f'--> Best solutions from the {hdf5_filename} will be written at: {best_file}')

    return sorted_by_chi2


def get_mcmc_results(hdf5_file, keywords=None, which_keys=False):
    """Extract the mcmc results from the hdf5 file. Returns a dictionary.

    Parameters
    ----------
    hdf5_file : str
        The hdf5 file from where the results will be taken
    keywords : list, optional
        A list of keywords in the hdf5 that will be returned. By default None,
        in which case all the keywords in hdf5 file are returned. In order to
        know which keywords are available, call which_keys=True
    which_keys : bool, optional
        A flag to print the available keywords in hdf5_file. Set to True to
        print the available keywords, by default False

    Returns
    -------
    dict
        A dictionary with the result of the specified keywords
    """
    
    f = h5py.File(hdf5_file, 'r')
    list_keys = list(f.keys())

    if which_keys:
        print(f"Available keywords: \n {list_keys}")
        f.close()
        return
    else:
        pass

    if keywords is not None:
        for keys in keywords:
            #assert(keys in f.keys()), 
            if keys not in f.keys():
                f.close()
                raise RuntimeError( f"Keyword -{keys}- does not exists in File."+\
                        f" Available keywords are : {list_keys}")
            else:
                pass
    else:
        keywords = f.keys()    

    output = {}
    for k in keywords:
        output[k] = f[k].value 
    f.close()

    return output


def gelman_rubin(chains=None, hdf5_file=None, nchunks_gr=10, thinning=1, names=None):
    """Perform the Gelman-Rubin test to assess for convergence of chains

    Parameters
    ----------
    chains : array, optional
        An array of the mcmc chains with shape (walkers,steps,dim), by default 
        None. If chains are given, then hdf5_file is ignored.
    hdf5_file : str, optional
        The hdf5 file name to extract the chains, by default None. 
    nchunks_gr : int, optional
        Number of chunks to divide the chains length, by default 10. At each
        node the Gelman-Rubin statistic is calculated.
    thinning : int, optional
        A factor to thin walkers, by default 1. Change to greater values for 
        longer chains or with several walkers.
    names : list, optional
        A list of names to match with the number of dimensions, by default None.
        These names will be returned in the dictionary.

    Returns
    -------
    dict
        A dictionary where keys are the names of the planet parameters if 
        'hdf5_file' is given, or those in the 'names' kwarg if 'chains' are 
        given. Values correspond to the Gelman-Rubin statistic at each node of 
        the nchunks_gr grid along the steps in the chain.
    """

    if chains is not None:
        assert(len(chains.shape)==3), "Shape for chains should be:"+\
            f" (walkers,steps,dim) instead of {chains.shape}"            

        if names is not None:
            pass
        else:
            # Create a generic list of names
            names = [f"dim{d}" for d in list(range(chains.shape[-1]))] 

    elif hdf5_file:
        f = h5py.File(hdf5_file, 'r')
        chains = f['CHAINS'].value[0,:,:,:]
        names = list(f['COL_NAMES'].value[:].split())
        f.close()

    else:
        raise RuntimeError("No chains or hdf5 file specified")

    nsteps= chains.shape[1]
    assert(nchunks_gr<nsteps), "nchunks_gr must be lower than "+\
        f"the number of saved steps: {nsteps}"

    
    # Select the steps to perform GR statistic 
    steps=[int(it) for it in np.linspace(0,nsteps,nchunks_gr+1)[:-1]]

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


def geweke(chains=None, hdf5_file=None, names=None, burnin=0.0):
    """Perform the Geweke test to assess for stationarity of the chains

    # Geweke criterion
    # https://rlhick.people.wm.edu/stories/bayesian_5.html
    # https://pymc-devs.github.io/pymc/modelchecking.html

    Parameters
    ----------
    chains : array, optional
        An array of the mcmc chains with shape (walkers,steps,dim), by default 
        None. If chains are given, then hdf5_file is ignored.
    hdf5_file : str, optional
        The hdf5 file name to extract the chains, by default None. 
    names : list, optional
        A list of names to match with the number of dimentions, by default None.
        These names will be returned in the dictionary.
    burnin : float, optional
        A fraction between 0 and 1 to discard as burn-in at the beggining of
        the chains, by default 0.0. This burnin will be applied over the
        array passed through 'chains', or over the chains extracted from the
        hdf5 file.

    Returns
    -------
    dict
        A dictionary where keys are the names of the planet parameters if 
        'hdf5_file' is given, or those in the 'names' kwarg if 'chains' are 
        given. Values correspond to the Z-score calculated along the 20 node at
        the second half of the chains (after burnin).
    """
    
    # Geweke criterion
    # https://rlhick.people.wm.edu/stories/bayesian_5.html
    # https://pymc-devs.github.io/pymc/modelchecking.html
    
    assert(0.0 <= burnin <= 1.0), f"burnin must be between 0 and 1!"

    if chains is not None:
        assert(len(chains.shape)==3), "Shape for chains should be:"+\
            f" (walkers,steps,dim) instead of {chains.shape}"
        ind = chains.shape[1] - 1

        if names is not None:
            pass
        else:
            # Create a generic list of names
            names = [f"dim{d}" for d in list(range(chains.shape[-1]))] 

    elif hdf5_file:
        f = h5py.File(hdf5_file, 'r')
        ind = f['INDEX'].value[0]
        chains = f['CHAINS'].value[0,:,:,:]
        names = list(f['COL_NAMES'].value[:].split())
        f.close()

    else:
        raise RuntimeError("No chains or hdf5 file specified")
    

    _burnin = int(burnin*ind)
    chains = chains[:,_burnin:ind+1,:]
    _ndim = chains.shape[-1]


    print("--> Performing Geweke test")

    # Convergence test over temperature 0
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
    for dimension in range(_ndim):
        ztas = []
        for sub20 in subsets_20:
            z = _z_score(subset_first_10[:,:,dimension], sub20[:,:,dimension])
            ztas.append(z)
        Z[names[dimension]] = ztas
    
    return Z


def _z_score(theta_a, theta_b):
    """A help function to calculate the Z-score used in the geweke test"""

    z = (np.mean(theta_a) - np.mean(theta_b)) / np.sqrt(np.var(theta_a) 
        + np.var(theta_b))
    
    return z


def intervals(frontiers, flat_params):
    """A function to probe wheter values are inside the stablished boundaries

    Parameters
    ----------
    frontiers : list
        A list of the boundaries with [min, max] values
    flat_params : list or array
        A list of flat parameters of same length as frontiers

    Returns
    -------
    list
        A list with boolean values. True's are for those parameters inside the 
        boundaries and False's for those outside. This list is returned as soon
        as a False is generated.
    """


    TF = []
    for i in range(len(flat_params)):
        if frontiers[i][0]<= flat_params[i] <= frontiers[i][1]:
            TF.append(True)
        else:
            TF.append(False)
            return TF
    return TF


def writefile(_file , writing , text, align):
    """A help function to write text in a file. Used in many functions"""

    txs = tuple(text.split())
    
    with open(_file, writing) as outfile:
        outfile.write(align % txs)
        outfile.close()

    return
    