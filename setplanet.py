from dataclasses import dataclass
import numpy as np
import sys

__doc__ = "Datos de setplanet"

__all__ = ["SetPlanet"]

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

    # Predefined boundaries. Change these physical bounds with responsibility 
    # and physical sense.
    physical_bounds = {
        'mass': [3e-4, 26635],  # Mearth
        'period': [0.1,500.0],  # days
        'ecc': [0.00001, 0.7],  
        'inclination': [0, 180.],  # deg 
        'argument': [0.0, 360.],  # deg
        'mean_anomaly': [0.0, 360.0],  # deg 
        'ascending_node': [0.0, 360.0]  # deg 
        }

    # Default boundaries are specified by physical_bounds
    mass  = physical_bounds["period"]
    period  = physical_bounds["period"]
    ecc  = physical_bounds["ecc"] 
    inclination = physical_bounds["inclination"]
    argument  = physical_bounds["argument"]
    mean_anomaly  = physical_bounds["mean_anomaly"] 
    ascending_node  = physical_bounds["ascending_node"]

    # TODO: Check the order of bounds are [lower_value, upper_value]
    # TODO: Que avise cuando se hace un corte físico
    def set_boundaries(self):


        self.boundaries = [ 
                    self._cut_boundaries("mass", self.mass),
                    self._cut_boundaries("period", self.period),
                    self._cut_boundaries("ecc", self.ecc),
                    self._cut_boundaries("inclination", self.inclination),
                    self._cut_boundaries("argument", self.argument),
                    self._cut_boundaries("mean_anomaly", self.mean_anomaly),
                    self._cut_boundaries("ascending_node", self.ascending_node)
                    ]
    
    def load_ttvs(self,ttvs_file):
        # TODO: Can I use pandas instead?
        """Add TTVs data to the planet        
        Arguments:
            ttvs_file {str} -- An ascci file containing transit number,
            transit time and lower and upper errors. Comments with #.
                For example:
                0 transit_time0 lower_error_0 upper_error_0
                1 transit_time1 lower_error_1 upper_error_1
                ...
        """
        ttvs_data = {}
        f = np.genfromtxt(f"{ttvs_file}", comments='#')
        for i in f:
            ttvs_data[int(i[0])] = [i[1], i[2], i[3] ]

        # Verify entry data are correct
        for ep, times in ttvs_data.items():
            if 0 in times:
                exit_status = "Invalid error values in epoch " + str(ep)+ \
                " found in file: " + str(ttvs_file)
                sys.exit(exit_status)
                
        self.ttvs_data = ttvs_data

        return

    def _cut_boundaries(self, param_str, param):
        """Stablish physical boundaries to each planetary parameter
        """
        phy_bds = self.physical_bounds[param_str]

        return [ max(phy_bds[0], param[0]), min(phy_bds[1], param[1]) ]