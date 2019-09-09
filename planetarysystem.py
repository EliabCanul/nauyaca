from operatettvs import Optimizers, MCMC
from plots import Plots
import numpy as np
from dataclasses import dataclass
from astropy import units as u
import sys
#import matplotlib.pyplot as plt
#from sklearn.linear_model import LinearRegression
#from utils import *


@dataclass
class PlanetarySystem(Optimizers, MCMC, Plots):
    """A planetary system is created when the stellar properties are set. Then
    planets are added and authomatically the number of planets increases. If
    planets have ttvs data, then these data are added to the TTVs dictionary.
    """

    #description = "Create a Planetary System"

    system_name : str
    mstar : float 
    rstar : float
    Ftime : float = 'auto'

    """
    def __init__(self, system_name, mstar, rstar):
        self.system_name = system_name
        self.mstar = mstar
        self.rstar = rstar
        self.all_planets = {}
        self.bounds = []
        self.planets_IDs = {}
        self.TTVs = {} 
        self.NPLA = 0
    """
    
    def add_planets(self, new_planets):
        parameters = ("mass", "period", "ecosw", "inclination", "esinw",
                    "mean_anomaly", "ascending_node")

        self.planets = {}
        self.bounds = []
        self.planets_IDs = {}
        self.TTVs = {} 
        self.NPLA = 0

        for new_planet in new_planets:
            # Dictionary with Planet objects
            self.planets[new_planet.planet_id] = new_planet
            
            # Create the flat boundaries
            for param in parameters:
                self.bounds.append(getattr(new_planet, param))

            # Dictionary that saves the entry order
            self.planets_IDs[new_planet.planet_id] = self.NPLA

            # Check for ttvs in planet object and append to TTVs dictionary
            if hasattr(new_planet, "ttvs_data"):
                self.TTVs[new_planet.planet_id] = new_planet.ttvs_data
            
            self.NPLA += 1

        self.calculate_constants()


    def calculate_constants(self):
        """NOTA: Tal vez esta funcion deberia estar en utils, y se haria una llamada 
        cada vez que se corra el optimizador o MCMC. Asi se podria quitar Ftime
        de esta clase y se colocaria aqui, con la finalidad de poder  modificar
        Ftime desde el script master
        --> calculate_constants(Ftime=200)
        Esto permitiria correr por ejemplo el optimizador por la mitad del 
        tiempo de los ttvs y asi seria mas rapido
        """

        autoFtime =  max([ list(self.TTVs[i].values())[-1][0] for i in \
                           self.TTVs.keys() ]) 

        # Set Ftime, total time of TTVs simulations [days]
        if str(self.Ftime).lower() == 'auto':
            self.Ftime = autoFtime
        if isinstance(self.Ftime, (int, float)):
            self.Ftime = self.Ftime
        else:
            raise Exception("Ftime must be int, float or option \"auto\" ")
        print('Total time of TTVs data: ', self.Ftime, ' [days]')

        # Discard TTVs outside the specified Ftime
        TTVs_copy = self.TTVs
        [[TTVs_copy[j].pop(i) for i in list(self.TTVs[j].keys()) \
            if self.TTVs[j][i][0]>self.Ftime] for j in list(self.TTVs.keys()) ]
        self.TTVs = TTVs_copy
        del TTVs_copy

        # Here I definie many other useful parameters
        tmp = [(i, self.TTVs[i][0][0]) for i in self.TTVs.keys()]

        # Detect which is the first planet in transit (return planetary ID)        
        self.first_planet_transit = min(tmp, key = lambda t: t[1])[0]

        # Detect which is the time of the first transit.
        self.T0JD = min(tmp, key = lambda t: t[1])[1]
        
        print('First planet (ID) in transit: ', self.first_planet_transit)
        print('Observed time of the first transit:', self.T0JD, ' [days]')

        # Make available rstar in AU units
        self.rstarAU = (self.rstar*u.Rsun).to(u.AU).value

        return



    # ---------------------------------------------------------------------

    """
    def nbody(self):
        pass
    """
    