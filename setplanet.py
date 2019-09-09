from dataclasses import dataclass
import matplotlib.pyplot as plt


class SetPlanet:
    """Every Planet has many properties as: 
    - planet_id (requiered)
    - boundaries for: mass, period, ...,
    - ttvs_data 
    """

    description = "A planet object"

    def __init__(self, planet_id):
        self.planet_id = planet_id 

    def boundaries(self, mass, period, ecosw, inclination, esinw,
                       mean_anomaly, ascending_node):

        self.mass = cut_boundaries("mass", mass)
        self.period = cut_boundaries("period", period)
        self.ecosw = cut_boundaries("ecosw", ecosw)
        self.inclination = cut_boundaries("inclination", inclination)
        self.esinw = cut_boundaries("esinw", esinw)
        self.mean_anomaly = cut_boundaries("mean_anomaly", mean_anomaly)
        self.ascending_node = cut_boundaries("ascending_node", ascending_node)


    def ttvs(self,ttvs_data):
        self.ttvs_data = ttvs_data






def cut_boundaries(param_str, param):

    max_msun = 1332000.
    physical_bounds = {
        'mass':[3e-4, max_msun], 
        'period':[0.001,500.0], 
        'ecosw':[-0.9999,0.9999],
        'inclination':[45.0, 135.0], 
        'esinw':[-0.9999,0.9999], 
        'mean_anomaly':[0.0,360.0], 
        'ascending_node':[0.0,360.0] }

    phy_bds = physical_bounds[param_str]

    return [ max(phy_bds[0], param[0]), min(phy_bds[1], param[1]) ]