import numpy as np
import ttvfast
from astropy import units as u
import sys

def run_TTVFast(flat_params, mstar=None, NPLA=None, Tin=0., Ftime=None):
    #Split 'flat_params' using 7 parameters per planet
    iters = [iter(flat_params)] * 7
    planets = list(zip(*iters))

    #Iteratively adds planet's parameters to TTVFast
    planets_list = []
    min_period = min([ p[1] for p in planets ])

    for planet in planets:
        #Be careful with the order!: m, per, e, inc, omega, M, Omega
        #See angle definitions: 
        # https://rebound.readthedocs.io/en/latest/python_api.html
        #TODO: refer to Nauyaca documentation (when exist)
        k = planet[2]
        h = planet[4]

        planets_list.append(
        ttvfast.models.Planet( 
            mass= (planet[0]*u.Mearth).to(u.Msun).value,
            period = planet[1], 
            eccentricity = np.sqrt( h**2 + k**2 ),
            inclination = planet[3],
            argument = np.rad2deg(np.arctan(h/k)),
            mean_anomaly = planet[5],
            longnode = planet[6]))

    signal = ttvfast.ttvfast(
            planets_list, stellar_mass=mstar, time=Tin, dt=min_period/100, 
            total=Ftime, rv_times=None, input_flag=1)   
    
    SP = signal['positions']

    try:
        lastidx = SP[2].index(-2.0)
        SP = [ i[:lastidx] for i in SP] #Remove array without data
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
        print('Warning: Invalid proposal')
        
    return EPOCHS    


def calculate_chi2(flat_params, self, flag="OPT"):

    Returns = {"OPT1": np.inf, "OPT2": 1e50, "OPT3": 1.0,
                "MCMC1": -np.inf, "MCMC2": -1e50, "MCMC3": -1.0}

    # Verify that proposal is inside boundaries
    inside_bounds = intervals(self, flat_params) 
    if False in inside_bounds:
        return Returns[flag+"1"]
    else:
        pass

    # Get 'positions' from signal in TTVFast
    signal_pos = run_TTVFast(
               flat_params, mstar=self.mstar, NPLA=self.NPLA, 
               Tin=self.T0JD, Ftime=self.Ftime+1)

    # Calculate transits epochs
    EPOCHS = calculate_epochs(signal_pos, self)

    #==== Compute chi^2
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
    POP0 = np.array(POP0).T.reshape(ntemps, nwalkers, self.NPLA*7)

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
            #print(mu, sig)
            return np.random.normal(loc=mu,scale=sig)
        
        if distr.lower() == 'picked':
            return np.random.choice(par)

    # Clean and sort results. Get just data inside threshold.
    [opt_data.remove(res) for res in opt_data if res[0]==1e50]
    opt_data.sort(key=lambda x: x[0])
    cut = int(len(opt_data)*threshold) - 1 #index of maximum chi2
    max_chi2 = opt_data[cut][0]
    [opt_data.remove(res) for res in opt_data if res[0] < max_chi2]
    param = np.array([x[1] for x in opt_data]).T

    POP0 = []
    for i, par in enumerate(param):
        poptmp=[]
        while len(poptmp) < ntemps*nwalkers:
            rdm = random_choice(distribution, par)
            if self.bounds[i][0] <= rdm and rdm <= self.bounds[i][1]:
                poptmp.append(rdm)

        POP0.append(poptmp)
    POP0 = np.array(POP0).T.reshape(ntemps, nwalkers, self.NPLA*7)
    
    return POP0


def logp(x):
    return 0.0


def intervals(self, flat_params):
    TF = []
    for i in range(len(flat_params)):
        if self.bounds[i][0]<= flat_params[i] <= self.bounds[i][1]:
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