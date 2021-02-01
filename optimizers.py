import time 
import datetime
import numpy as np  # Se puede sustituir np.random.seed()?
import sys
from dataclasses import dataclass
from multiprocessing import Pool
from .utils import * # Miztli: from NAU.utils import *
from .utils import writefile, intervals, _chunks, cube_to_physical, _remove_constants, calculate_chi2 # Miztli: from NAU.utils import writefile, intervals
import copy
import warnings
from scipy.optimize import differential_evolution
from scipy.optimize import minimize
warnings.filterwarnings("ignore")


__doc__ = "Optimizers"

__all__ = ["Optimizers"]




# ------------------------ O P T I M I Z E R ------------------------------

#f = lambda x: x-360 if x>360 else (360+x if x<0 else x)

@dataclass
class Optimizers:

    """Fit the TTVs running sequentially the algorithms:
    * Differential evolution 
    * Powell
    * Nelder Mead
    """

    PSystem : None
    nsols : int = 1
    cores : int = 1  
    path : str = './'
    suffix : str = ''

    @staticmethod
    def _differential_evolution_nelder_mead(PSystem, base_name, indi=0):
        
        base_name_cube, base_name_phys = base_name
        ##hypercube = [[0.,1.] for _ in range(len(PSystem.bounds)) ] # hypercube
        
        """ 
                        -----Differential Evolution----- 
        """
        # From doc: To improve your chances of finding a global minimum 
        # use higher popsize values, with higher mutation and (dithering), 
        # but lower recombination values
        DE = differential_evolution( 
                calculate_chi2, PSystem.hypercube, #PSystem.bounds,  # hypercube
                disp=False, seed=np.random.seed(),
                popsize=100, maxiter=5, polish=False,
                mutation=(1.5,1.9), recombination=0.2,
                args= (PSystem,))
        x0 = DE.x
        f0 = DE.fun

        """
                            -----Powell-----
        """
        PO = minimize(
            calculate_chi2, list(x0), method= 'Powell', 
            bounds=PSystem.hypercube, #PSystem.bounds,  # hypercube
            options={'maxiter':15000, 'maxfev':15000, 
                     'xtol':0.000001, 'ftol':0.1, 'disp':False, 'adaptive':True},
            args=(PSystem,))
        x1 = PO.x
        f1 = PO.fun

        """ 
                            -----Nealder-Mead-----
        """
        NM = minimize(
            calculate_chi2, list(x1), method= 'Nelder-Mead', 
            options={'maxiter':15000, 'maxfev':15000,
                     'xatol':0.000001, 'fatol':0.1, 'disp':False, 'adaptive':True},
            args=(PSystem,))
        x2 = NM.x
        f2 = NM.fun
        
        # Verbose
        print(f" {indi+1} | DE: {f0 :.3f}  PW: {f1 :.3f}  NM: {f2 :.3f}")

        # Reconstruct flat_params adding the constant values
        x2_cube = x2.tolist() #list(x2)  
        #for k, v in PSystem.constant_params.items(): 
        #    x2_cube.insert(k, v)

        # Write results in output file
        info = '{} \t {} \n'.format(str(f2), "  ".join([str(it ) for it in x2_cube]))
        
        writefile( base_name_cube, 'a', info, '%-30s'
                        +' %-11s'*(len(info.split())-1) + '\n')


        #---------------- 
        # Convert from cube to physical values and remove constants
        x2_phys = cube_to_physical(PSystem, x2)

        x2_phys = _remove_constants(PSystem, x2_phys).tolist()

        # Write results in output file
        #info = '{} \t {} \n'.format(str(f2), "  ".join([str(it ) for it in x2_phys]))
        info = ' ' + str(f2) + ' ' + "  ".join([str(it ) for it in x2_phys]) 
        writefile( base_name_phys, 'a', info, '%-30s'
                        +' %-11s'*(len(info.split())-1) + '\n')

        #---------------- 
        # Return cube and physical results
        #return [f2] + list(x2_cube)
        return [f2 , x2_cube, x2_phys]


    #@classmethod
    def run(self):
        """[summary]
        
        Arguments:
            PSystem {[type]} -- [description]
        
        Keyword Arguments:
            nsols {int} -- Number of solutions to be performed (default: 1)
            cores {int} -- Number of cores to run in parallel (default: 1)
            suffix {str} -- Suffix of the outpu file. (default: '')
        
        Returns:
            array -- An array with the chi**2 and the planetary solutions for 
            each planet. Also an ascci file is saved with the same information.
        """

        ta = time.time()
        now = datetime.datetime.now()

        print( "\n =========== OPTIMIZATION ===========\n")
        print( "--> Starting date: ", now.strftime("%Y-%m-%d %H:%M"))
        print(f"--> Finding {self.nsols} solutions using {self.cores} cores")

        # Output file name
        #base_name = '{}'.format(PSystem.system_name) + suffix + ".opt"
        base_name_cube = f'{self.path}{self.PSystem.system_name}_cube' + self.suffix + ".opt"
        base_name_phys = f'{self.path}{self.PSystem.system_name}_phys' + self.suffix + ".opt"

        # Write headers in both files
        header = '#Chi2 ' + " ".join([i for i in self.PSystem.params_names.split()])
        
        writefile(base_name_cube, "w", header, '%-15s'+' %-11s'*(len(header.split())-1)+'\n')
        writefile(base_name_phys, "w", header, '%-15s'+' %-11s'*(len(header.split())-1)+'\n')

        # Verbose
        print("--> Results will be saved at:")
        print(f'     * {base_name_cube} (normalized)')
        print(f'     * {base_name_phys} (physical)')
        print(f'--> Reference time of the solutions: {self.PSystem.T0JD} [days]')
        print('- - - - - - - - - - - - - - - - - - - - -')
        print('Solution  |   chi square from optimizers'        )
        print('- - - - - - - - - - - - - - - - - - - - -')

        # Run in parallel
        with warnings.catch_warnings(record=True) as w: 
            warnings.simplefilter("always")

            pool = Pool(processes=self.cores)
            results = [pool.apply_async(self._differential_evolution_nelder_mead,
                        args=(self.PSystem, [base_name_cube,base_name_phys], i)) 
                        for i in range(self.nsols)]
            output = [p.get() for p in results]		
        pool.terminate()
        print(f'Elapsed time in optimization: {(time.time() - ta)/60 :.3f} minutes')

        # Sort results by chi2
        res = self.sort_results(output)

        res_complete = {}
        res_complete['chi2'] = np.array([r[0] for r  in res])
        res_complete['cube'] = np.array([r[1] for r  in res])
        res_complete['physical'] = np.array([r[2] for r  in res])

        self.results = res_complete

        return res_complete 


    @staticmethod
    def sort_results(X):
        return sorted(X, key=lambda x:x[0])

    
    @property
    def results_dict(self):
        """Return physical results sorted by chi2 and for individual planet parameters"""
        
        self.results_by_param = dict(zip(self.PSystem.params_names.split(), self.results['physical'].T))
        self.results_by_param['chi2'] = self.results['chi2']
        
        return self.results_by_param