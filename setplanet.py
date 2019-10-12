from dataclasses import dataclass
import matplotlib.pyplot as plt
import numpy as np
import copy

@dataclass
class SetPlanet:
    """ 
    Defines a new Planet object.
    Each Planet has many properties as: 
    - planet_id (requiered)
    - boundaries for: mass, period, ...,
    - ttvs_data given as dictionary 
    """

    planet_id : str

    def set_boundaries(self, mass, period, ecc, inclination, argument,
                       mean_anomaly, ascending_node):

        self.boundaries = [ 
                    _cut_boundaries("mass", mass),
                    _cut_boundaries("period", period),
                    _cut_boundaries("ecc", ecc),
                    _cut_boundaries("inclination", inclination),
                    _cut_boundaries("argument", argument),
                    _cut_boundaries("mean_anomaly", mean_anomaly),
                    _cut_boundaries("ascending_node", ascending_node)
                    ]

        return


    def add_ttvs(self,ttvs_data):
        """Add TTVs data to the planet
        
        Arguments:
            ttvs_data {dict} -- A dictionary containing transit number as
                key and a list of transit times with errors as values.
                For example:
                {0:[transit_time0, lower_error_e0, upper_error_E0],
                 1:[transit_time1, lower_error_e1, upper_error_E1], ...}
        """
        self.ttvs_data = ttvs_data

        return

def _cut_boundaries(param_str, param):
    """Stablish physical boundaries to each planetary parameter
    """
    max_msun = 1332000. #Maximum planet mass in Earth masses
    physical_bounds = {
        'mass':[3e-4, max_msun],  # Mearth
        'period':[0.001,500.0],  # days
        'ecc': [0.0001, 0.7],  #
        'inclination': [45, 135.],  # deg 
        'argument': [0., 360.],  # deg
        'mean_anomaly':[0.0, 360.],  # deg 
        'ascending_node':[0.0, 360.]  # deg 
        }

    phy_bds = physical_bounds[param_str]

    return [ max(phy_bds[0], param[0]), min(phy_bds[1], param[1]) ]
