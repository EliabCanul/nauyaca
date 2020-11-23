from dataclasses import dataclass
from collections import OrderedDict
import numpy as np
import copy
import pickle

__doc__ = "PS"

__all__ = ["PlanetarySystem"]

col_names = ["mass", "period", "ecc", "inclination", "argument", "mean_anomaly",
             "ascending_node"]

units = ["[M_earth]", "[d]", "", "[deg]", "[deg]", "[deg]", "[deg]"]

Msun_to_Mearth = 332946.07832806994

@dataclass
class PlanetarySystem:  
    """A Planetary System object is created when the stellar properties are set.
    Then planets are added and authomatically the number of planets increases. 
    If planets have ttvs data, then these data are added to the TTVs dictionary.
    Inputs:
    Name of the planetary system
    Stellar mass [Msun]
    Stellar radius [Rsun]
    Time interval to take from the TTVs data [days]. By "default" it takes the entire
     time of the TTVs.
    """

    system_name : str           
    mstar : float                
    rstar : float               
    Ftime : float = "Default" 
    dt : float = 0.1   

    def add_planets(self, new_planets):

        self.planets = {}
        self.bounds = []
        self._bounds_parameterized = []
        self.planets_IDs = {}
        self.TTVs = {} 
        self.NPLA = 0

        for new_planet in new_planets:
            # Dictionary with Planet objects
            self.planets[new_planet.planet_id] = new_planet
            
            # Upper mass boundary should be at most a small fracc of mstar
            _check_mass_limits(self.mstar, new_planet)

            # Create flat boundaries
            self.bounds.extend(new_planet.boundaries)

            # Create parameterized flat boundaries
            self._bounds_parameterized.extend(new_planet._boundaries_parameterized)

            # Dictionary that saves the entry order
            self.planets_IDs[new_planet.planet_id] = self.NPLA

            # Check for ttvs in planet object and append to TTVs dictionary
            # TODO: checar si se puede cambiar a new_planet.ttvs_data.copy()
            # y quitar el import de copy
            if hasattr(new_planet, "ttvs_data"): 
                self.TTVs[new_planet.planet_id] = copy.deepcopy(
                                                new_planet.ttvs_data)
            
            self.NPLA += 1

        # Calculate necessary constants and parameters
        _calculate_constants(self, self.Ftime)

        # Save the planetary system object using pickle
        pickle_file = f'{self.system_name}.pkl'
        with open(pickle_file, 'wb') as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)
        print(f"--> Pickle file {pickle_file} has been saved")

        return

    def __str__(self):
        print("\n =========== Planetary System Summary =========== ")
        summary = [f"System: {self.system_name}"]
        summary.append(f"Mstar: {self.mstar} Msun |  Rstar: {self.rstar} Rsun")
        if self.NPLA > 0:
            summary.append(f"Number of planets: {self.NPLA}")
            summary.append(f"Planet information:")
            for k, v in self.planets.items():
                summary.append("------"+"\n"+f"Planet: {k}")
                summary.append("  Boundaries:")
                summary.append("\n".join([f"    {col_names[i]}: {str(bo)}  {units[i]}" for i,bo in enumerate(v.boundaries)] ))
                if hasattr(self.planets[k], "ttvs_data"):
                    summary.append("  TTVs: True")
                else:
                    summary.append("  TTVs: False")
            summary.append(f"\nTotal time of TTVs data: {self.Ftime} [days]")
            summary.append(f"First planet (ID) in transit: {self.first_planet_transit}")
            summary.append(f"Reference epoch of the solutions: {self.T0JD} [JD]")
            summary.append(f"Time span of TTVs simulations: {self.time_span}")
            summary.append(f"Timestep of the simulations: {self.dt} [days]")

        return "\n".join(summary)


def _check_mass_limits(ms, Planet):
    
    upper_mass_limit = Planet.boundaries[0][1]    
    k_limit = 0.01 # Percentage of stellar mass to set as upper mass bound

    if upper_mass_limit > (k_limit * ms) * Msun_to_Mearth:
        new_limit = (k_limit * ms) * Msun_to_Mearth
        Planet.boundaries[0][1] = new_limit 
        print(f'Upper mass boundary have been set to {new_limit} [Mearth] for planet {Planet.planet_id}')

    return


def _calculate_constants(PSystem, Ftime):
    
    assert len(PSystem.TTVs) > 0, "No Transit times have been provided for any planet. \
        Use .load_ttvs() attribute in Planet object."

    # Identify constant parameters and save them
    # TODO: change constant_params from dict with numbers as keys for parameter strings?
    PSystem.constant_params = OrderedDict() #{}
    for iPSb in range(len(PSystem.bounds)):
        if float(PSystem.bounds[iPSb][0]) == float(PSystem.bounds[iPSb][1]):
            PSystem.constant_params[iPSb] = float(PSystem.bounds[iPSb][0])
    ##PSystem = sorted(PSystem.constant_params.items(), key=lambda j:j[0])
    
    # Remove these boundaries of constants values from bounds
    indexes_remove = list(PSystem.constant_params.keys())
    for index in sorted(indexes_remove, reverse=True):
        del PSystem.bounds[index]
        # Nuevo. Delete constant values in parameterized bounds
        del PSystem._bounds_parameterized[index]
        # Nuevo

    PSystem.ndim = len(PSystem.bounds) # Number of dimensions

    # Nuevo
    # Create an hypercube with bounds 0-1
    PSystem.hypercube = [[0.,1.] for _ in range(len(PSystem.bounds)) ]
    PSystem.bi, PSystem.bf = np.array(PSystem._bounds_parameterized).T 
    # Nuevo

    # Create a string with parameter names
    params_names = []
    for i in range(1, PSystem.NPLA+1):
        for c in col_names:
            params_names.append(c+f"{i}")
    PSystem.params_names_opt = "  ".join(params_names)

    for index in sorted(indexes_remove, reverse=True):
        del params_names[index]
    params_names = "  ".join(params_names)
    PSystem.params_names = params_names

    # Set Ftime, total time of TTVs simulations [days]
    if str(Ftime).lower() == 'default':
        PSystem.Ftime = max([ list(PSystem.TTVs[i].values())[-1][0] for i in \
                    PSystem.TTVs.keys() ]) 

    elif isinstance(Ftime, (int, float)):
        PSystem.Ftime = Ftime
    
    else:
        print(PSystem.Ftime)
        raise Exception("Ftime must be int, float or option \"Default\" ")

    # Discard TTVs outside the specified Ftime
    # TODO: se puede usar .copy()?
    TTVs_copy = PSystem.TTVs
    [[TTVs_copy[j].pop(i) for i in list(PSystem.TTVs[j].keys()) \
        if PSystem.TTVs[j][i][0]>PSystem.Ftime] for j in list(PSystem.TTVs.keys()) ]
    PSystem.TTVs = TTVs_copy
    del TTVs_copy

    # Set the maximum number of transits
    ##PSystem.n_transits = {p: max(PSystem.TTVs[p].keys()) for p in PSystem.TTVs.keys()}
    
    # Create dictionary of mean transit erros and the constant part of the 
    # likelihood function.
    PSystem.sigma_obs = {}
    for plnt_id, ttvs_obs in PSystem.TTVs.items():
        mean_error = {}
        for epoch, times in ttvs_obs.items():
            mean_error[epoch] = (times[1] + times[2])/2.
        PSystem.sigma_obs[plnt_id] = mean_error

    PSystem.second_term_logL = {}
    for plnt_id, d_errs in PSystem.sigma_obs.items():
        arg_log = 2*np.pi* np.array(list(d_errs.values()))
        PSystem.second_term_logL[plnt_id] = np.sum( 0.5*np.log(arg_log) )
        #
        #arg_log = 2*np.sqrt(2.)* np.array(list(d_errs.values()))
        #PSystem.second_term_logL[plnt_id] = np.sum( np.log(1./arg_log) )


    # Detect which is the first planet in transit (return planetary ID)        
    # FIXME: Does is it a necessary parameter?
    tmp = []
    for k, v in PSystem.TTVs.items():
        t0_index = list(sorted(PSystem.TTVs[k]))[0]
        tmp.append( (k, PSystem.TTVs[k][t0_index][0] ) )
    PSystem.first_planet_transit = min(tmp, key = lambda t: t[1])[0]


    # Detect which is the time of the first transit and round it down.
    # --->>> T0JD will be the reference epoch of the solutions <<<---
    # FIXME: Parche hecho para que T0JD sea 0 para los sintéticos y min para
    # los sistemas reales
    if PSystem.system_name.startswith('syn'):
        PSystem.T0JD = 0.0
    else:
        PSystem.T0JD =  ( min(tmp, key = lambda t: t[1])[1] ) // 1
    
    # Time span to make the simulations (DO NOT STABLISH IT MANUALLY!)
    PSystem.time_span = 1.005*(PSystem.Ftime - PSystem.T0JD )#+ 0.1*(PSystem.T0JD) #+2.5 #quitar el 1

    # Make available rstar in AU 
    Rsun_to_AU = 0.004650467260962157
    PSystem.rstarAU = PSystem.rstar*Rsun_to_AU 

    return
