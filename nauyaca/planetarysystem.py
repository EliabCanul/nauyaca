from dataclasses import dataclass
from collections import OrderedDict
from .constants import col_names, units, Msun_to_Mearth, Rsun_to_AU, k_limit
import numpy as np
import warnings
import pickle
import json
import copy
import sys


__doc__ = "A module to create Planetary System objects over which simulations will be performed"

__all__ = ["PlanetarySystem"]


# TODO: In the future, check for compatibility between dataclasses and slots
# in order to avoid boiler plate code
#@dataclass
class PlanetarySystem:  
    """A Planetary System object formed by a star and planets
    
    A Planetary System object is created when the stellar properties are given
    and Planets are added.

    """

    # All instances of the class
    __slots__ = ('system_name',
                    'mstar',
                    'rstar',
                    't0',
                    'ftime',
                    'dt',
                    'planets',
                    'bounds',
                    '_bounds_parameterized',
                    'planets_IDs',
                    'TTVs',
                    '_TTVs_original',
                    'npla',
                    'constant_params',
                    'params_names_all',
                    'params_names',
                    'ndim',
                    'hypercube',
                    'bi',
                    'bf',
                    'transit_times',
                    'sigma_obs',
                    'second_term_logL',
                    'rstarAU',
                    'valid_t0'
                    )

    def __init__(self, system_name,mstar,rstar):
        """
        Parameters
        ----------
        system_name : str
            The Planetary System name
        mstar : float
            Stellar mass [Msun]
        rstar : float
            Stellar radius [Rsun]
        """        

        self.system_name = system_name
        self.mstar = mstar
        self.rstar = rstar


    def add_planets(self, new_planets):
        """A function to add Planets to the Planetary System

        Parameters
        ----------
        new_planets : list
            A list with Planet objects that will be part of the system.
            Recommendable: Add Planet objects in order of closeness to the star
        """

        self.planets = {}
        self.bounds = []
        self._bounds_parameterized = []
        self.planets_IDs = {}  # OrderedDict?
        self._TTVs_original = {} 
        self.npla = 0 

        for new_planet in new_planets:
            # Dictionary with Planet objects
            self.planets[new_planet.planet_id] = new_planet
            
            # Upper mass boundary should be at most a fraction of mstar
            self._check_mass_limits(new_planet)

            # Create flat boundaries. Calls boundaries attribute (property) 
            self.bounds.extend(new_planet.boundaries)

            # Create parameterized flat boundaries
            self._bounds_parameterized.extend(new_planet._boundaries_parameterized)

            # Dictionary that saves the entry order
            self.planets_IDs[new_planet.planet_id] = self.npla

            # Check for ttvs in planet object and append to TTVs dictionary
            if hasattr(new_planet, "ttvs_data") and type(new_planet.ttvs_data) == dict: 
                self._TTVs_original[new_planet.planet_id] = new_planet.ttvs_data.copy()
            
            self.npla += 1

        # Calculate necessary attributes of Planetary System objects
        # from Planets objects added above.
        self._set_mandatory_attr_()


    def _set_mandatory_attr_(self):
        """A help function to manage attributes of the system
        """
        
        assert len(self._TTVs_original) > 0, "No Transit times have been provided for any "\
            "planet. Use .load_ttvs() method to load an ASCII file or use the "\
            ".ttvs_data attribute in the Planet object to load an dictionary with "\
            "transit ephemerids in format: "\
            "\n{epoch0: [time0, lower_e0, upper_e0], epoch1: [time1, lower_e1, upper_e1], ...}"

        # Attributes of parameter space
        self._manage_boundaries()

        # Attributes of simulations
        self.simulation()

        return


    def _manage_boundaries(self):
        """A function to adapt Planet boundaries to Planetary System boundaries
        """

        # Identify constant parameters and save them
        # TODO: change constant_params from dict with numbers as keys for parameter strings?
        self.constant_params = OrderedDict() #{}
        for iPSb in range(len(self.bounds)):
            if float(self.bounds[iPSb][0]) == float(self.bounds[iPSb][1]):
                self.constant_params[iPSb] = float(self.bounds[iPSb][0])
        ##self = sorted(self.constant_params.items(), key=lambda j:j[0])
        
        # Remove these boundaries of constants values from bounds
        indexes_remove = list(self.constant_params.keys())
        for index in sorted(indexes_remove, reverse=True):
            del self.bounds[index]
            # Nuevo. Delete constant values in parameterized bounds
            del self._bounds_parameterized[index]
            # Nuevo

        # Create a string with parameter names
        params_names = []
        for i in range(1, self.npla+1):
            for c in col_names:
                params_names.append(c+f"{i}")
        self.params_names_all = "  ".join(params_names)

        for index in sorted(indexes_remove, reverse=True):
            del params_names[index]
        params_names = "  ".join(params_names)
        self.params_names = params_names

        # Ascending node of at least one Planet must be fixed
        o_keys = list(self.constant_params.keys())
        cparams = [self.params_names_all.split()[o_k] for o_k in o_keys]
        assert sum([cp.startswith("ascending_node") for cp in cparams]) > 0, "The"\
            " ascending node of at least one planet must be fixed to avoid degenerated solutions."\
            " \nSet the lower and upper boundaries of a Planet as equal,"\
            " for example, Planet1.ascending_node = [90,90] "


        # Number of dimensions
        self.ndim = len(self.bounds)

        return


    def simulation(self, t0=None, ftime=None, dt=None):
        """A function to set the simulation features

        Parameters
        ----------
        t0 : float, optional
            Time of reference for the simulation results (days), 
            by default None, in which case, t0 is calculated by rounding 
            down the smallest transit time in Planet ephemeris.
        ftime : float, optional
            The final time of the simulations (days), by default None, 
            in which case, ftime is calculated by rounding up the maximum 
            transit time in the Planet ephemeris.
        dt : float, optional
            The timestep of the simulations (days), by default None, 
            in which case, dt is calculated as the estimated internal 
            planet period divided over 30.
        """

        # Make available rstar in AU         
        self.rstarAU = self.rstar*Rsun_to_AU 

        # Create an hypercube with bounds 0-1.
        # hypercube is used to rescale the different parameters to be dimensionless
        # it's useful to stablish convergence in optimizers
        self.hypercube = [[0., 1.]]*self.ndim 

        ##self.bi, self.bf = np.array(self._bounds_parameterized).T 
        # Converted to lists to be json serializable
        self.bi, self.bf = list(map(list, zip(*self._bounds_parameterized)))

        # Adapt TTVs to t0 and ftime specifications. Make a copy
        self.TTVs = copy.deepcopy(self._TTVs_original)

        # Calculates constants to manage the reference time
        first_transits = []
        first_transit_epochs = []
        estimated_periods = []
        first_line = []
        for k in self.TTVs.keys():
            first_epoch = list(sorted(self.TTVs[k]))[0]
            first_transit_epochs.append(first_epoch)
            first_transits.append( (k, self.TTVs[k][first_epoch][0]) )

            TT = [x[0] for x in list(self.TTVs[k].values())]
            tmp_periods = []
            # Estimates planet periods based in provided transit times     
            for t in range(len(TT)-1):
                tmp_periods.append((TT[t+1]-TT[t]))
            estimated_periods.append(min(tmp_periods))

            # A tuple of data to calculate min_t0
            first_line.append((first_epoch, self.TTVs[k][first_epoch][0], min(tmp_periods)))

        # Detect the smaller central time of the first planet in transit 
        first_central_time = min(first_transits, key = lambda t: t[1])[1]

        # IMPORTANT:
        # Verify that there is at least one epoch labeled with 0!
        # Otherwise it is impossible to set attributes for simulations
        # since TTVFast returns epochs starting with 0 and it will be
        # impossible to calculate O-C properly.
        if 0 in first_transit_epochs:
            pass
        else:
            exit_status = "TTVs ephemeris do not have at least one transit epoch "+\
                "labeled with 0. Relabel the ephemeris transit epochs to match " +\
                "the first transit with epoch 0. The first transit must occur "+\
                "after t0. See documentation for details."
            sys.exit(exit_status)

        # Estimate the lower possible value for t0
        ficticional_transit = [] # It would correspond to transit epoch -1
        for epi, tti, peri in first_line:
            while epi>-1:
                epi -= 1
                tti -= peri
            ficticional_transit.append(tti)

        min_t0 = max(ficticional_transit) 
        
        # Manage timestep
        if dt == None:
            # Estimate dt based on the lower estimated period
            self.dt = min(estimated_periods)/30.
        elif isinstance(dt, (int,float)):
            # Set given dt
            self.dt = dt
        else:
            raise ValueError("Invalid timestep -dt-")
        
        # Stablish t0
        if isinstance(t0, (int, float)):
            self.t0 = t0
        else:
            # Detect which is the time of the first transit and round it down.
            self.t0 =  ( min(first_transits, key = lambda t: t[1])[1] ) // 1

        # Detect if the proposed reference time t0 is valid
        self.valid_t0 = (min_t0, first_central_time)
        if min_t0 < self.t0 < first_central_time:
            pass
        else:
            raise ValueError(f"-t0- must be greater than {min_t0} and " +\
                f"lower than the first transit time {first_central_time}, " +\
                "to avoid spurious results. ")

        # Set ftime: total time of TTVs simulations [days]
        if ftime == None: 
            # Takes the maximum central time in TTVs and round it up to the next integer
            self.ftime = np.ceil(max([ list(self.TTVs[i].values())[-1][0] for i in \
                        self.TTVs.keys() ]) )

        elif isinstance(ftime, (int, float)):
            self.ftime = ftime

        else:
            print(ftime)
            raise ValueError("-ftime- must be int, float or None. "+
            "In the last case (None), ftime is automatically chosen.  ")


        # Discard TTVs outside the specified ftime
        TTVs_copy = copy.deepcopy(self.TTVs)
        [[TTVs_copy[j].pop(i) for i in list(self.TTVs[j].keys()) \
            if self.TTVs[j][i][0]>self.ftime] for j in list(self.TTVs.keys()) ]
        self.TTVs = TTVs_copy
        del TTVs_copy

        # Create dictionary of central transit times and mean transit errors
        self.sigma_obs = {}
        self.transit_times = {}
        for planet_id, ttvs_obs in self.TTVs.items():
            mean_error = {}
            timing = {}
            for epoch, times in ttvs_obs.items():
                timing[epoch] = times[0]
                mean_error[epoch] = (times[1] + times[2])/2.
            self.sigma_obs[planet_id] = mean_error
            self.transit_times[planet_id] = timing

        # Create dictionary of the constant part of the likelihood function.
        self.second_term_logL = {}
        for planet_id, d_errs in self.sigma_obs.items():
            arg_log = 2*np.pi* np.array(list(d_errs.values()))**2
            self.second_term_logL[planet_id] = np.sum( 0.5*np.log(arg_log) )


        return


    def _check_mass_limits(self, Planet):
        """A help function to check for upper limits in mass. Maximum planet
        mass must be, at most, -k_limit- of the stellar mass. k_limit is
        specified in constants.
        """
        
        upper_mass_limit = Planet.mass[1]    

        k_mass_frac = (k_limit * self.mstar) * Msun_to_Mearth

        if upper_mass_limit > k_mass_frac:
            Planet.mass = (Planet.mass[0], k_mass_frac)  # Update mass limits
            Planet.boundaries  # Update planet boundaries
            print(f'--> Upper mass boundary for planet -{Planet.planet_id}- has been'+
            f' set to {k_limit*100}% of stellar mass: {k_mass_frac} [Mearth]')        
        else:
            pass

        return


    def save_pickle(self,  path='./'):
        """Save the Planetary System object using pickle

        Parameters
        ----------
        path : str, optional
            A path to save the pickle file, by default './'
        """

        pickle_file = path+f'{self.system_name}.pkl'
        with open(pickle_file, 'wb') as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)
        print(f"--> Pickle file {pickle_file} saved")

        return


    @staticmethod
    def load_pickle(pickle_file):
        """A function to rebuild the Planetary System from pickle

        Parameters
        ----------
        pickle_file : str
            The file name of the .pkl file

        Returns
        -------
        Planetary System
            Returns the Planetary System rebuilded
        """

        with open(f'{pickle_file}', 'rb') as input:
            pickle_object = pickle.load(input)
        
        return pickle_object


    def save_json(self, path='./'):
        """Save the Planetary System object using json

        Parameters
        ----------
        path : str, optional
            A path to save the json file, by default './'
        """
        
        # Serialize planet objects
        dct_planet_obj = {}
        for planetID in self.planets_IDs:

            dct_attr = {}
            # Iterable of planet attributes:
            for pl_attr in self.planets[planetID].__slots__: 
                try:
                    dct_attr[pl_attr] = getattr(self.planets[planetID], pl_attr)
                except:
                    print(f"Unable to serialize {pl_attr}")

            dct_planet_obj[planetID] = dct_attr

        # Serialize planetary system
        dct_ps_attr = {}
        # Iterable of planetary system attributes:
        for ps_attr in self.__slots__:
            try:
                if ps_attr == 'planets':
                    dct_ps_attr[ps_attr] = dct_planet_obj
                else:
                    dct_ps_attr[ps_attr] = getattr(self, ps_attr)
            except:
                pass

        json_file = path+f'{self.system_name}.json'
        with open(f'{json_file}', 'w') as outputfile:
            json.dump(dct_ps_attr, outputfile, indent=4)
        
        print(f"--> JSON file {json_file} saved")

        return


    @classmethod
    def load_json(cls, json_file):
        """A function to rebuild the Planetary System from json

        Parameters
        ----------
        json_file : str
            The file name of the .json file

        Returns
        -------
        Planetary System
            Returns the Planetary System rebuilded
        """

        try:
            from .setplanet import SetPlanet
        except:
            print("Unable to import SetPlanet")

        with open(f"{json_file}", "r") as read_file:
            json_load = json.load(read_file)

        # Initialize Planetary System        
        new_PS = PlanetarySystem(system_name=json_load['system_name'],
                                rstar=json_load['rstar'],
                                mstar=json_load['mstar'])
 
        
        # Add Planet instances to list
        planet_list = []
        for pl_attr in json_load['planets'].values():

            new_planet = SetPlanet(pl_attr['planet_id'])

            new_planet.mass = tuple(pl_attr['mass'])
            new_planet.period = tuple(pl_attr['period'])
            new_planet.ecc = tuple(pl_attr['ecc'])
            new_planet.inclination = tuple(pl_attr['inclination'])
            new_planet.argument = tuple(pl_attr['argument'])
            new_planet.mean_anomaly = tuple(pl_attr['mean_anomaly'])
            new_planet.ascending_node = tuple(pl_attr['ascending_node'])

            individual_ttvs = pl_attr['ttvs_data']
            individual_ttvs = {int(k):v for k,v in individual_ttvs.items()}

            new_planet.ttvs_data = individual_ttvs 

            # append planet objects
            planet_list.append(new_planet)

        # Add the decoded planets
        new_PS.add_planets(planet_list)

        # Restart the simulation attributes.
        # If more attributes exist, include them here!
        new_PS.simulation(t0=json_load['t0'],
                          ftime=json_load['ftime'],
                          dt=json_load['dt'])

        return new_PS       

        
    def __str__(self):
        """Prints a summary of the Planetary System object

        Returns
        -------
        str
            A summary of the Planetary System object
        """

        print("\n =========== Planetary System Summary =========== ")
        summary = [f"\nSystem: {self.system_name}"]
        summary.append(f"Mstar: {self.mstar} Msun |  Rstar: {self.rstar} Rsun")
        if hasattr(self, 'npla'):
        #if self.npla > 0:
            summary.append(f"Number of planets: {self.npla}")
            summary.append(f"Planet information:")
            for k, v in self.planets.items():
                number = self.planets_IDs[v.planet_id] + 1
                summary.append("------"+"\n"+f"Planet{number}: {k}")
                summary.append("  Boundaries:")
                summary.append("\n".join([f"    {col_names[i]}: {str(bo)}  {units[i]}" 
                                        for i,bo in enumerate(v.boundaries)] ))
                if hasattr(self.planets[k], "ttvs_data") and type(self.planets[k].ttvs_data) == dict:
                    summary.append("  TTVs: True")
                else:
                    summary.append("  TTVs: False")
            summary.append("\nSimulation attributes: ")
            ##summary.append(f"\nFirst planet (ID) in transit: {self.first_planet_transit}")
            summary.append(f"Reference epoch of the solutions (t0): {self.t0} [JD]")
            summary.append(f"Total time of TTVs data (ftime): {self.ftime} [days]")
            ##summary.append(f"Time span of TTVs simulations: {self.time_span}")
            summary.append(f"Timestep of the simulations (dt): {self.dt} [days]")

        return "\n".join(summary)
