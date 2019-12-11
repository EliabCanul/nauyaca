from operatettvs import Optimizers, MCMC
import numpy as np
from dataclasses import dataclass
import sys
import copy
import pickle

col_names = ["mass", "period", "ecc", "inclination", "argument", "mean_anomaly",
             "ascending_node"]

@dataclass
class PlanetarySystem:  
    """A Planetary System object is created when the stellar properties are set.
    Then planets are added and authomatically the number of planets increases. 
    If planets have ttvs data, then these data are added to the TTVs dictionary.
    """

    system_name : str
    mstar : float 
    rstar : float
    Ftime : float = "Default"
    
    print("\nWELCOME TO NAUYACA: A TOOL FOR THE TTVs INVERSION PROBLEM\n")

    def add_planets(self, new_planets):

        self.planets = {}
        self.bounds = []
        self.planets_IDs = {}
        self.TTVs = {} 
        self.NPLA = 0

        for new_planet in new_planets:
            # Dictionary with Planet objects
            self.planets[new_planet.planet_id] = new_planet
            
            # Create the flat boundaries
            self.bounds.extend(new_planet.boundaries)

            # Dictionary that saves the entry order
            self.planets_IDs[new_planet.planet_id] = self.NPLA

            # Check for ttvs in planet object and append to TTVs dictionary
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
            for k, v in self.planets.items():
                summary.append("------"+"\n"+f"{k}")
                summary.append("     "+"      ".join(col_names))
                summary.append("  "+"  ".join([ str(i) for i in v.boundaries] ))
                if hasattr(self.planets[k], "ttvs_data"):
                    summary.append("  TTVs: True")
                else:
                    summary.append("  TTVs: False")
            summary.append(f"\nTotal time of TTVs data: {self.Ftime} [days]")
            summary.append(f"First planet (ID) in transit: {self.first_planet_transit}")
            summary.append(f"Observed time of the first transit: {self.T0JD} [days]")
            summary.append(f"Interval time to make the simulations: {self.sim_interval}")

        return "\n".join(summary)

def _calculate_constants(PSystem, Ftime):

    # Identify constant parameters and save them
    PSystem.constant_params = {}
    for iPSb in range(len(PSystem.bounds)):
        if float(PSystem.bounds[iPSb][0]) == float(PSystem.bounds[iPSb][1]):
            PSystem.constant_params[iPSb] = float(PSystem.bounds[iPSb][0])
    
    # Remove these boundaries of constants values from bounds
    indexes = list(PSystem.constant_params.keys())
    for index in sorted(indexes, reverse=True):
        del PSystem.bounds[index]

    # Create a string with parameter names
    params_names = []
    for i in range(1, PSystem.NPLA+1):
        for c in col_names:
            params_names.append(c+f"{i}")
    
    for index in sorted(indexes, reverse=True):
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
    TTVs_copy = PSystem.TTVs
    [[TTVs_copy[j].pop(i) for i in list(PSystem.TTVs[j].keys()) \
        if PSystem.TTVs[j][i][0]>PSystem.Ftime] for j in list(PSystem.TTVs.keys()) ]
    PSystem.TTVs = TTVs_copy
    del TTVs_copy

    # Detect which is the first planet in transit (return planetary ID)        
    tmp = []
    for k, v in PSystem.TTVs.items():
        t0_index = list(sorted(PSystem.TTVs[k]))[0]
        tmp.append( (k, PSystem.TTVs[k][t0_index][0] ) )
    PSystem.first_planet_transit = min(tmp, key = lambda t: t[1])[0]

    # Detect what is the time of the first transit.
    PSystem.T0JD = min(tmp, key = lambda t: t[1])[1]

    # Time's window to make the simulations
    PSystem.sim_interval = PSystem.Ftime - PSystem.T0JD + 0.1*(PSystem.T0JD)

    # Make available rstar in AU units
    Rsun_to_AU = 0.004650467260962157
    PSystem.rstarAU = PSystem.rstar*Rsun_to_AU #u.Rsun).to(u.AU).value

    return
