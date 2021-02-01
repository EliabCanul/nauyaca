from dataclasses import dataclass
from collections import OrderedDict
from .constants import col_names, units, Msun_to_Mearth, Rsun_to_AU
import numpy as np
import pickle
import json
import copy

__doc__ = "PS"

__all__ = ["PlanetarySystem"]

#! col_names = ["mass", "period", "ecc", "inclination", "argument", "mean_anomaly",
#             "ascending_node"]

#! units = ["[M_earth]", "[d]", "", "[deg]", "[deg]", "[deg]", "[deg]"]


# TODO: In the future, check for compatibility between dataclasses and slots
# in order to avoid boiler plate code

#@dataclass
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

    # All instances of the class
    __slots__ = ('system_name',
                    'mstar',
                    'rstar',
                    'Ftime',
                    'dt',
                    'planets',
                    'bounds',
                    '_bounds_parameterized',
                    'planets_IDs',
                    'TTVs',
                    '_TTVs_original',
                    'NPLA',
                    'constant_params',
                    ###'params_names_opt',
                    'params_names',
                    'ndim',
                    'hypercube',
                    'bi',
                    'bf',
                    'transit_times',
                    'sigma_obs',
                    'second_term_logL',
                    ##'first_planet_transit',
                    'T0JD',
                    ##'time_span',
                    'rstarAU',
                    )

    def __init__(self, system_name,mstar,rstar):#,Ftime='default',dt=None):
        self.system_name = system_name
        self.mstar = mstar
        self.rstar = rstar
        #self.NPLA = 0
        #self.Ftime = 'Default'
        #self.dt = None


    """
    # Dataclass format. Unused still dataclass supports slots
    system_name : str           
    mstar : float                
    rstar : float               
    Ftime : float = "Default" 
    dt : float = 0.1   
    """
    def add_planets(self, new_planets):

        self.planets = {}
        self.bounds = []
        self._bounds_parameterized = []
        self.planets_IDs = {}
        self._TTVs_original = {} 
        self.NPLA = 0 

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
            self.planets_IDs[new_planet.planet_id] = self.NPLA

            # Check for ttvs in planet object and append to TTVs dictionary
            if type(new_planet.ttvs_data) == dict: 
                self._TTVs_original[new_planet.planet_id] = new_planet.ttvs_data.copy()
            
            self.NPLA += 1

        # Calculate necessary attributes of Planetary System objects
        # from Planets objects added above.
        self._set_mandatory_attr_()


    def _set_mandatory_attr_(self):
        
        assert len(self._TTVs_original) > 0, "No Transit times have been provided for any "\
            "planet. Use .load_ttvs() method to load an ASCII file or use the "\
            ".ttvs_data attribute in Planet object to load an dictionary with "\
            "transit ephemerids in format: "\
            "\n{epoch0: [time0, lower_e0, upper_e0],epoch1: [time1, lower_e1, upper_e1]}"

        # Attributes of parameter space
        self._manage_boundaries()

        # Attributes of simulations
        self.simulation()

        return


    def _manage_boundaries(self):
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
        for i in range(1, self.NPLA+1):
            for c in col_names:
                params_names.append(c+f"{i}")
        ###self.params_names_opt = "  ".join(params_names)

        for index in sorted(indexes_remove, reverse=True):
            del params_names[index]
        params_names = "  ".join(params_names)
        self.params_names = params_names

        # Number of dimensions
        self.ndim = len(self.bounds)

        return


    #def _simulation_attributes_(self):
    def simulation(self, T0JD=None, Ftime='Default', dt=None):

        # Make available rstar in AU         
        self.rstarAU = self.rstar*Rsun_to_AU 

        # Create an hypercube with bounds 0-1
        self.hypercube = [[0., 1.]]*self.ndim 

        ##self.bi, self.bf = np.array(self._bounds_parameterized).T 
        # Converted to lists to be json serializable
        self.bi, self.bf = list(map(list, zip(*self._bounds_parameterized)))

        # Adapt TTVs to T0JD and Ftime specifications. Make a copy
        self.TTVs = copy.deepcopy(self._TTVs_original)

        
        # Calculates constants to manage the reference time
        first_transits = []
        estimated_periods = []
        for k in self.TTVs.keys():
            first_epoch = list(sorted(self.TTVs[k]))[0]
            first_transits.append( (k, self.TTVs[k][first_epoch][0]) )

            TT = [x[0] for x in list(self.TTVs[k].values())]
            tmp_periods = []
            # Estimates planet periods based in provided transit times     
            for t in range(len(TT)-1):
                tmp_periods.append((TT[t+1]-TT[t]))
            estimated_periods.append(min(tmp_periods))

        ## Detect which is the first planet in transit (return planetary ID)
        #self.first_planet_transit = min(first_transits, key = lambda t: t[1])[0]
        # Detect the smaller central time of the first planet in transit 
        first_central_time = min(first_transits, key = lambda t: t[1])[1]
        # Estimate the lower possible value for T0JD
        min_t0 = first_central_time - min(estimated_periods)
        
        # Manage timestep
        if dt == None:
            # Estimate dt based on the lower estimated period
            self.dt = min(estimated_periods)/30.
        elif isinstance(dt, (int,float)):
            # Set given dt
            self.dt = dt
        else:
            raise ValueError("Invalid timestep -dt-")
        
        # Detect which is the time of the first transit and round it down.
        # --->>> T0JD will be the reference epoch of the solutions <<<---
        if self.system_name.startswith('syn'):
            # FIXME: Parche hecho para que T0JD sea 0 para los sintéticos y min para
            # los sistemas reales            
            self.T0JD = 0.0

        elif isinstance(T0JD, (int, float)):
            self.T0JD = T0JD

        else:
            self.T0JD =  ( min(first_transits, key = lambda t: t[1])[1] ) // 1


        # Detect if the proposed reference time is valid
        if min_t0 < self.T0JD < first_central_time:
            pass

        else:
            raise ValueError(f"-T0JD- must be lower than the first transit " +\
                f"time: {first_central_time}, but greater than {min_t0} to " +\
                "avoid spurious results")

        ## Time span to make the simulations (DO NOT SET IT MANUALLY!)
        ##self.time_span = 1.005*(self.Ftime - self.T0JD )#+ 0.1*(self.T0JD)

        # Set Ftime: total time of TTVs simulations [days]
        if str(Ftime).lower() == 'default':
            # Takes the maximum central time in TTVs and round it up to the next integer
            self.Ftime = np.ceil(max([ list(self.TTVs[i].values())[-1][0] for i in \
                        self.TTVs.keys() ]) )

        elif isinstance(Ftime, (int, float)):
            #self.Ftime = self.Ftime
            self.Ftime = Ftime

        else:
            #print(self.Ftime)
            print(Ftime)
            raise ValueError("-Ftime- must be int, float or option \"Default\" ")


        # Discard TTVs outside the specified Ftime
        # TODO: se puede usar .copy()?
        TTVs_copy = copy.deepcopy(self.TTVs)
        [[TTVs_copy[j].pop(i) for i in list(self.TTVs[j].keys()) \
            if self.TTVs[j][i][0]>self.Ftime] for j in list(self.TTVs.keys()) ]
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
            #arg_log = np.log(1./(np.sqrt(2*np.pi)*np.array(list(d_errs.values())) ))
            #self.second_term_logL[planet_id] = sum(arg_log)
            #
            #arg_log = 2*np.sqrt(2.)* np.array(list(d_errs.values()))
            #self.second_term_logL[planet_id] = np.sum( np.log(1./arg_log) )

        return


    def _check_mass_limits(self, Planet):
        
        #! upper_mass_limit = Planet.boundaries[0][1]
        upper_mass_limit = Planet.mass[1]    
        # Percentage of stellar mass to set as upper mass limit
        k_limit = 0.01 
        k_mass_frac = (k_limit * self.mstar) * Msun_to_Mearth

        if upper_mass_limit > k_mass_frac:
            #! new_limit = (k_limit * ms) * Msun_to_Mearth
            #! Planet.boundaries[0][1] = new_limit
            #! Planet.boundaries = Planet.boundaries._replace(mass = (Planet.mass[0],new_limit) )
            Planet.mass = (Planet.mass[0], k_mass_frac)  # Update mass limits
            Planet.boundaries  # Update planet boundaries
            print(f'--> Upper mass boundary for planet -{Planet.planet_id}- has been'+
            f' set to {k_limit*100}% of stellar mass: {k_mass_frac} [Mearth]')        
        else:
            pass

        return

    # ========== Saving ==========
    @property
    def save_pickle(self):
        """ Save the planetary system object using pickle"""
        pickle_file = f'{self.system_name}.pkl'
        with open(pickle_file, 'wb') as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)
        print(f"--> Pickle file {pickle_file} saved")

        return


    #@property
    #def save(self):  # save_pickle?
    #    # Save pickle file with Planetary System object
    #    self._save_pickle()
    #    return 


    @staticmethod
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


    @property
    def save_json(self):
        
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

        json_file = f'{self.system_name}.json'
        with open(f'{json_file}', 'w') as outputfile:
            json.dump(dct_ps_attr, outputfile, indent=4)
        
        print(f"--> JSON file {json_file} saved")

        return


    #@property
    #def save_json(self):
    #    self._encoder_json()
    #    return


    @classmethod
    def load_json(cls, json_file):
        """ Create a Planetary System instance from json.
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
                                #Ftime=json_load['Ftime'],
                                #dt=json_load['dt'])    
        
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
        new_PS.simulation(T0JD=json_load['T0JD'],
                          Ftime=json_load['Ftime'],
                          dt=json_load['dt'])

        return new_PS       

        
    def __str__(self):
        print("\n =========== Planetary System Summary =========== ")
        summary = [f"\nSystem: {self.system_name}"]
        summary.append(f"Mstar: {self.mstar} Msun |  Rstar: {self.rstar} Rsun")
        if hasattr(self, 'NPLA'):
        #if self.NPLA > 0:
            summary.append(f"Number of planets: {self.NPLA}")
            summary.append(f"Planet information:")
            for k, v in self.planets.items():
                summary.append("------"+"\n"+f"Planet: {k}")
                summary.append("  Boundaries:")
                summary.append("\n".join([f"    {col_names[i]}: {str(bo)}  {units[i]}" 
                                        for i,bo in enumerate(v.boundaries)] ))
                if type(self.planets[k].ttvs_data) == dict: #hasattr(self.planets[k], "ttvs_data"):
                    summary.append("  TTVs: True")
                else:
                    summary.append("  TTVs: False")
            summary.append("\nSimulation attributes: ")
            ##summary.append(f"\nFirst planet (ID) in transit: {self.first_planet_transit}")
            summary.append(f"Reference epoch of the solutions (T0JD): {self.T0JD} [JD]")
            summary.append(f"Total time of TTVs data (Ftime): {self.Ftime} [days]")
            ##summary.append(f"Time span of TTVs simulations: {self.time_span}")
            summary.append(f"Timestep of the simulations (dt): {self.dt} [days]")

        return "\n".join(summary)
