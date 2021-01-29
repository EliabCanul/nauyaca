

# var_names
col_names = (
			"mass", 
			"period", 
			"ecc", 
			"inclination", 
			"argument", 
			"mean_anomaly",
            "ascending_node"
            )
			
units = ("[M_earth]", "[d]", " ", "[deg]", "[deg]", "[deg]", "[deg]")


# Predefined boundaries. Change these physical bounds with responsibility 
# and physical sense.
physical_bounds = {
    'mass': (0.0123032, 25426.27252757),  #  (lower=1 moon ;upper limit = 80 Mjup ) Mearth
    'period': (0.1, 1000.0),  # days
    'ecc': (0.000001, 0.9),  
    'inclination': (0.0, 180.),  # deg 
    'argument': (0.0, 360),  # deg 
    'mean_anomaly': (0.0, 360),  # deg 
    'ascending_node': (0.0, 360.0)  # deg 
    }

Msun_to_Mearth = 332946.07832806994

Rsun_to_AU = 0.004650467260962157


colors = {0:"red", 
          1:"olive", 
          2:"skyblue", 
          3:"gold", 
          4:"teal",
          5:"orange", 
          6:"purple"}


labels = {0:r"$\mathrm{Mass}$" ,
           1:r"$\mathrm{Period}$",
           2:r"$\mathrm{eccentricity}$",
           3:r"$\mathrm{Inclination}$",
           4:r"$\mathrm{Argument\ (\omega)}$",
           5:r"$\mathrm{Mean\ anomaly}$",
           6:r"$\mathrm{\Omega}$"}

units_latex = {0:r"$\mathrm{[M_{\oplus}]}$",
         1:r"$\mathrm{[days]}$",
         2:r"$\mathrm{ }$",
         3:r"$\mathrm{[deg]}$",
         4:r"$\mathrm{[deg]}$",
         5:r"$\mathrm{[deg]}$",
         6:r"$\mathrm{[deg]}$"}
