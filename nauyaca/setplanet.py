import sys
from dataclasses import dataclass
import collections
from .constants import col_names, units, physical_bounds


__doc__ = "A module to create Planet objects and their attributes"

__all__ = ["SetPlanet"]


# TODO: In the future, check for compatibility between dataclasses and slots  
# in order to avoid boiler plate code
#@dataclass
class SetPlanet:
    """Set a new Planet object.

    Planet object contains the main information of individual planets
    that will be part of a Planetary System object.

    """

    # All instances of the class
    __slots__ = ('planet_id', 
                'mass', 
                'period', 
                'ecc', 
                'inclination', 
                'argument', 
                'mean_anomaly', 
                'ascending_node', 
                '_boundaries_parameterized',
                'ttvs_data')


    def __init__(self, planet_id):
        """
        Parameters
        ----------
        planet_id : str
            The Planet name
        """

        # Default boundaries are specified by physical_bounds
        # Initialize default boundaries
        self.planet_id = planet_id
        self.mass = physical_bounds["mass"]  # mass
        self.period = physical_bounds["period"] # period
        self.ecc = physical_bounds["ecc"] # eccentricity
        self.inclination = physical_bounds["inclination"] # inclination
        self.argument = physical_bounds["argument"] # argument of periastron
        self.mean_anomaly = physical_bounds["mean_anomaly"] # mean_anomaly
        self.ascending_node = physical_bounds["ascending_node"] # ascending_node
        #self._boundaries_parameterized = _boundaries_parameterized
        self.ttvs_data = None


    @property
    def boundaries(self):
        """Set the planetary boundaries of the Planet.

        Valid parameters are: mass, period, ecc, inclination, argument,
        mean_anomaly, ascending_node.

        Returns
        -------
        namedtuple
            The boundaries of each planet parameter with format [lower, upper]
        """
      
        Boundaries = collections.namedtuple('Boundaries', col_names) 

        # Physical boundaries
        bounds = Boundaries(
            mass = self._cut_boundaries("mass", self.mass),
            period = self._cut_boundaries("period", self.period),
            ecc = self._cut_boundaries("ecc", self.ecc),
            inclination = self._cut_boundaries("inclination", self.inclination),
            argument = self._cut_boundaries("argument", self.argument),
            mean_anomaly = self._cut_boundaries("mean_anomaly", self.mean_anomaly),
            ascending_node = self._cut_boundaries("ascending_node", self.ascending_node)
            )

        # Parameterized boundaries. Substitute argument (w) and mean anomaly (M)
        # for parameterized angles x=w+M and y=w-M.
        # It is done to deal with the periodic boundaries of these angles.
        # _boundaries_parameterized follows the same order as in col_names.
        self._boundaries_parameterized = (
            bounds[0],
            bounds[1],
            bounds[2],
            bounds[3],
            (min(bounds[4])+min(bounds[5]), max(bounds[4])+max(bounds[5])),
            (min(bounds[4])-max(bounds[5]), max(bounds[4])-min(bounds[5])),
            bounds[6]
            )
        
        return bounds

    
    def load_ttvs(self, ttvs_file, comment="#", delimiter=None):
        """Reads an ASCII file with the transit ephemeris

        The format of the input file must be:
        EPOCH  TIME  TIME_LOWER_ERR  TIME_UPPER_ERR

        where epoch must be a integer and times must be in days. Epoch 0 must be
        referenced to a reference time t0.

        Parameters
        ----------
        ttvs_file : str
            The file name with transit ephemeris
        comment : str, optional
            Symbol to take as comments, by default "#"
        delimiter : str, optional
            Symbol to take as delimiter between columns, by default None,
            in which case, white spaces are taken as delimiters
        """

        ttvs_data = {}

        f = open(f"{ttvs_file}",'r').read().splitlines()
        
        lines = [line for line in f if line]
        for line in lines:
            
            if line.startswith(f"{comment}"):
                pass
            else:
                l = line.split(delimiter)
                ttvs_data[int(l[0])] = [float(l[1]), float(l[2]), float(l[3]) ]   

        # Verify entry data are correct
        for ep, times in ttvs_data.items():
            # Verify uncertainties are not 0
            if 0 in [times[1], times[2]]:
                self.ttvs_data = None
                exit_status = "Invalid uncertainties in epoch " + str(ep)+ \
                " found in file: " + str(ttvs_file)
                sys.exit(exit_status)
                
        self.ttvs_data = ttvs_data

        return


    def _cut_boundaries(self, param_str, param):
        """Verifies planet boundaries

        Parameter boundaries definied manually outside the established 
        physical boundaries are truncated to the lower or upper limits.

        Parameters
        ----------
        param_str : str
            Planet parameter in string
        param : class attribute
            Planet parameter as class attribute

        Returns
        -------
        tuple
            The truncated boundary (if outside) or the current boundary
        """

        assert param[0]<=param[1], f"{param_str}:{param} lower bound is " \
        "greater than upper. Boundaries must be: lower <= upper"

        phy_bds = physical_bounds[param_str]

        lower = max(phy_bds[0], param[0])
        upper = min(phy_bds[1], param[1])

        idx_p = col_names.index(param_str)

        if param[0] < phy_bds[0]:
            pass
            print(f"--> Parameter -{param_str}- of planet -{self.planet_id}- set to lower physical limit: {phy_bds[0]} {units[idx_p]}")
        if param[1] > phy_bds[1]:
            pass
            print(f"--> Parameter -{param_str}- of planet -{self.planet_id}- set to upper physical limit: {phy_bds[1]} {units[idx_p]}")

        return (lower,  upper)


    def __str__(self):
        """Prints a summary of the Planet object

        Returns
        -------
        str
            A summary of the Planet object
        """

        print("\n =========== Planet Summary =========== ")
        summary = [f"Planet : {self.planet_id}"]
        summary.append("  Boundaries:")
        for i, v in enumerate(self.boundaries):
            summary.append("\n".join([f"    {col_names[i]}: {v}  {units[i]}" ] ) )
        if hasattr(self, "ttvs_data") and type(self.ttvs_data) == dict:
            summary.append("  TTVs: True")
        else:
            summary.append("  TTVs: False")
        return "\n".join(summary)
        