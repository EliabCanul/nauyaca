import sys
import copy
import warnings
import time 
import datetime
import numpy as np
from scipy.optimize import differential_evolution
from scipy.optimize import minimize
from dataclasses import dataclass
from multiprocessing import Pool
from .utils import * 
from .utils import writefile, intervals, _chunks, cube_to_physical, _remove_constants, calculate_chi2 

warnings.filterwarnings("ignore")


__doc__ = "A module to run a sequence of minimization algorithms"

__all__ = ["Optimizers"]


@dataclass
class Optimizers:
    """Simulations based on a sequence of minimization algorithms
        Differential Evolution -> Powell ->  Nelder-Mead
    
    Parameters
    ----------
    PSystem : 
        The Planetary System object
    nsols : int, optional
        The number of solutions to reach, by default 1
    cores : int, optional
        The number of cores to run in parallel, by default 1
    path : str, optional
        The directory path to save outputs, by default './'
    suffix : str, optional
        A suffix for the output file name, by default ''
    """

    PSystem : None
    nsols : int = 1
    cores : int = 1  
    path : str = './'
    suffix : str = ''


    @staticmethod
    def DE_PW_NM(PSystem, base_name, indi=0):
        """Base function to run the sequence of minimization algorithms:

        Parameters
        ----------
        PSystem : Planetary System
            The Planetary System object to simulate
        base_name : tuple
            A tuple with the output file names for the 'cube' and 'phys' results
        indi : int, optional
            An ID that refers to number of solution, by default 0

        Returns
        -------
        list
            A list containing: [fun, X_cube, X_phys], where:
            fun: chi square of the solution
            X_cube : Solution in normalized bounds
            X_phys : Solution in physical values
        """                
        
        base_name_cube, base_name_phys = base_name
        
        """ 
                        -----Differential Evolution----- 
        """
        # From doc: To improve your chances of finding a global minimum 
        # use higher popsize values, with higher mutation and (dithering), 
        # but lower recombination values
        DE = differential_evolution( 
                calculate_chi2, PSystem.hypercube, #PSystem.bounds,  # hypercube
                disp=False, seed=np.random.seed(),
                popsize=30, maxiter=1, polish=False,
                mutation=(1.2,1.7), recombination=0.6,
                args= (PSystem,))
        x0 = DE.x
        f0 = DE.fun
        
        # Reduce the boundaries after DE such that PO and NM explore a smaller 
        # parameter space. It would help to avoid getting solutions stucked 
        # in the current boundaries. Modify momentarily the hypercube 
        PSystem.hypercube = [ [max(0,x-.3), min(1,x+.3)] for x in x0]
        
        """
                            -----Powell-----
        """
        PO = minimize(
            calculate_chi2, list(x0), method= 'Powell', 
            #bounds=PSystem.hypercube, #PSystem.bounds,  # Unnecessary in recent scipy versions
            options={'maxiter':12000, 'maxfev':12000, 
                     'xtol':0.001, 'ftol':1, 'disp':False},
            args=(PSystem,))
        x1 = PO.x
        f1 = PO.fun

        """ 
                            -----Nealder-Mead-----
        """
        NM = minimize(
            calculate_chi2, list(x1), method= 'Nelder-Mead', 
            options={'maxiter':15000, 'maxfev':15000,
                     'xatol':0.0001, 'fatol':1, 'disp':False, 'adaptive':True},
            args=(PSystem,))
        x2 = NM.x
        f2 = NM.fun

        # Verbose
        print(f" {indi+1} |  {f0 :.3f} --> {f1 :.3f} --> {f2 :.3f}  |  {f2/PSystem.dof['total'] :.3f}")

        # Reconstruct flat_params adding the constant values
        x2_cube = x2.tolist() 

        # Write results in output file
        info = '{} \t {} \n'.format(str(f2), "  ".join([str(it ) for it in x2_cube]))
        
        writefile( base_name_cube, 'a', info, '%-30s'
                        +' %-11s'*(len(info.split())-1) + '\n')

        #---------------- 
        # Convert from cube to physical values and remove constants
        x2_phys = cube_to_physical(PSystem, x2)

        x2_phys = _remove_constants(PSystem, x2_phys).tolist()

        # Write results in output file
        info = ' ' + str(f2) + ' ' + "  ".join([str(it ) for it in x2_phys]) 
        writefile( base_name_phys, 'a', info, '%-30s'
                        +' %-11s'*(len(info.split())-1) + '\n')


        return [f2 , x2_cube, x2_phys]


    def run(self):
        """Call this function to run the optimizers

        Returns
        -------
        dict
            A dictionary with the optimization results. Keys are:
            'chi2' : The corresponding chi square of the solutions
            'cube' : The solutions normalized between 0 and 1
            'physical' : The solutions in physical values
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
        print(f'--> Reference time of the solutions: {self.PSystem.t0} [days]')
        print('- - - - - - - - - - - - - - - - - - - - -')
        print('Solution  |  chi square minimization sequence  |  Reduced chi square'        )
        print('- - - - - - - - - - - - - - - - - - - - -')

        # Run in parallel
        with warnings.catch_warnings(record=True) as w: 
            warnings.simplefilter("always")

            pool = Pool(processes=self.cores, maxtasksperchild=1) # Helps to liberate memory!
            results = [pool.apply_async(self.DE_PW_NM,
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
        """A function to order the solutions according to chi2

        Parameters
        ----------
        X : list
            A list containing lists of the solutions, where the first item of
            individual lists is the corresponding chi square

        Returns
        -------
        list
            Returns a list of lists with chi square in increasing order
        """
        return sorted(X, key=lambda x:x[0])

    
    @property
    def results_dict(self):
        """A function to separate the solutions by individual planet parameters

        Returns
        -------
        dict
            Returns physical results sorted by chi2 and for individual planet 
            parameters
        """
        
        self.results_by_param = dict(zip(self.PSystem.params_names.split(), self.results['physical'].T))
        self.results_by_param['chi2'] = self.results['chi2']

        return self.results_by_param
