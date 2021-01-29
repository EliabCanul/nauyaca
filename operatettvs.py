#import multiprocessing as mp
import time 
import datetime
import numpy as np
import sys
from dataclasses import dataclass
from multiprocessing import Pool
from .utils import * # Miztli: from NAU.utils import *
from .utils import writefile, intervals, _chunks, cube_to_physical, _remove_constants # Miztli: from NAU.utils import writefile, intervals
import copy
import warnings
from scipy.optimize import differential_evolution
from scipy.optimize import minimize
warnings.filterwarnings("ignore")

import h5py
import ptemcee as pt
from contextlib import closing

__doc__ = "MCMC + NM"

__all__ = ["Optimizers", "MCMC"]

# TODO: Output files should be saved at specified directory



# ------------------------ O P T I M I Z E R ------------------------------

f = lambda x: x-360 if x>360 else (360+x if x<0 else x)
@dataclass
class Optimizers:

    """Fit the TTVs running sequentially the algorithms:
    * Differential evolution 
    * Powell
    * Nelder Mead
    """

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
                log_likelihood_func, PSystem.hypercube, #PSystem.bounds,  # hypercube
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
            log_likelihood_func, list(x0), method= 'Powell', 
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
            log_likelihood_func, list(x1), method= 'Nelder-Mead', 
            options={'maxiter':15000, 'maxfev':15000,
                     'xatol':0.000001, 'fatol':0.1, 'disp':False, 'adaptive':True},
            args=(PSystem,))
        x2 = NM.x
        f2 = NM.fun
        
        # Verbose
        print(f" {indi+1} | DE: {f0 :.3f}  PO: {f1 :.3f}  NM: {f2 :.3f}")

        # Reconstruct flat_params adding the constant values
        x2_cube = list(x2)  
        #for k, v in PSystem.constant_params.items(): 
        #    x2_cube.insert(k, v)

        # Write results in output file
        info = '{} \t {} \n'.format(str(f2), "  ".join([str(it ) for it in x2_cube]))
        
        writefile( base_name_cube, 'a', info, '%-30s'
                        +' %-11s'*(len(info.split())-1) + '\n')


        #---------------- 
        # Convert from cube to physical values and remove constants
        x2_phys = cube_to_physical(PSystem, x2)

        x2_phys = _remove_constants(PSystem, x2_phys)

        # Write results in output file
        #info = '{} \t {} \n'.format(str(f2), "  ".join([str(it ) for it in x2_phys]))
        info = ' ' + str(f2) + ' ' + "  ".join([str(it ) for it in x2_phys]) 
        writefile( base_name_phys, 'a', info, '%-30s'
                        +' %-11s'*(len(info.split())-1) + '\n')

        #---------------- 

        return ([f2] + list(x2_cube))


    @classmethod
    def run_optimizers(cls, PSystem,
                        nsols=1,
                        cores=1,  
                        suffix=''):
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
        print(f"--> Finding {nsols} solutions using {cores} cores")

        # Output file name
        #base_name = '{}'.format(PSystem.system_name) + suffix + ".opt"
        base_name_cube = f'{PSystem.system_name}_cube' + suffix + ".opt"
        base_name_phys = f'{PSystem.system_name}_phys' + suffix + ".opt"

        # Write headers in both files
        header = '#Chi2 ' + " ".join([i for i in PSystem.params_names.split()])
        
        writefile(base_name_cube, "w", header, '%-15s'+' %-11s'*(len(header.split())-1)+'\n')
        writefile(base_name_phys, "w", header, '%-15s'+' %-11s'*(len(header.split())-1)+'\n')

        # Verbose
        print("--> Results will be saved at:")
        print(f'     * {base_name_cube} (normalized)')
        print(f'     * {base_name_phys} (physical)')
        print('- - - - - - - - - - - - - - - - - - - - -')
        print('Solution  |   chi square from optimizers'        )
        print('- - - - - - - - - - - - - - - - - - - - -')

        # Run in parallel
        with warnings.catch_warnings(record=True) as w: 
            warnings.simplefilter("always")

            pool = Pool(processes=cores)
            results = [pool.apply_async(cls._differential_evolution_nelder_mead,
                        args=(PSystem, [base_name_cube,base_name_phys], i)) for i in range(nsols)]
            output = [p.get() for p in results]		
        pool.terminate()
        print(f'Time elapsed in optimization: {(time.time() - ta)/60 :.3f} minutes')

        return output        



# ------------- M A R K O V  C H A I N  M O N T E - C A R L O -------------

@dataclass
class MCMC:

    """Perform an MCMC using Parallel-Tempering"""

    @classmethod
    def run_mcmc(cls, PSystem, 
                Itmax=100, # run_time ??
                conver_steps=2, 
                cores=1,
                nwalkers=None, 
                ntemps=None, 
                Tmax=None, 
                betas=None, 
                pop0=None, 
                verbose = True,
                suffix=''): 
        # FIXME: does is it necessary nwalkers, ntemps? or it can be determined
        # from pop0
        """
        Run the parallel-tempering algorithm
        """

        if ntemps is not None:
            cls.ntemps = ntemps
        if betas is not None:
            cls.ntemps = len(betas)

        cls.nwalkers = nwalkers
        cls.Itmax = Itmax
        cls.conver_steps = conver_steps
        cls.cores = cores
        cls.betas = betas
        cls.Tmax = Tmax
        cls.pop0 = pop0

        ###ndim = len(PSystem.bounds) # Number of dimensions
        if cls.nwalkers < PSystem.ndim:
            sys.exit(f"Number of walkers must be >= 2*ndim, i.e., \
                nwalkers = {2*PSystem.NPLA*7}.\n Stopped simulation!")

        ti = time.time()
        now = datetime.datetime.now()
        print("\n =========== PARALLEL-TEMPERING MCMC ===========\n")
        print("--> Starting date: ", now.strftime("%Y-%m-%d %H:%M"))

        print("--> Reference epoch of the solutions: ", PSystem.T0JD, " [JD]")
        # TODO: print a summary of the input parameters for MCMC!
        """ 
                Create and set the hdf5 file for saving the output.
        """
        
        # hdf5 file name to save mcmc data
        # TODO: hdf5_filename should be by default the name of the planetary 
        # system, but also another name can be used. Set an file_name keyword?
        # It's usefull for restarting mcmc
        cls.hdf5_filename = f"{PSystem.system_name}{suffix}.hdf5"
        print('--> Results will be saved at: ', cls.hdf5_filename, '\n')
        
        # Create an h5py file
        cls._set_hdf5(cls, PSystem, cls.hdf5_filename)

        """
                Run the MCMC using parallel-tempering algorithm
        """
        # Default values in ptemcee, 
        # Do not change it at least you have read Vousden et al. (2016):
        nu = 100. #/nwalkers 
        t0 = 10000. #/nwalkers
        a_scale = 10

        with closing(Pool(processes=cls.cores)) as pool:

            sampler = pt.Sampler(
                                nwalkers=cls.nwalkers,
                                dim=PSystem.ndim,
                                logp=cls._logp, 
                                logl=log_likelihood_func,
                                ntemps=cls.ntemps,
                                betas=cls.betas,
                                adaptation_lag = t0,
                                adaptation_time=nu,
                                a=a_scale, 
                                Tmax=cls.Tmax,
                                pool=pool,
                                loglargs=(PSystem,), 
                                logpkwargs={'psystem':PSystem},
                                loglkwargs={'flag':'MCMC'})

            index = 0
            autocorr = np.empty( cls.nsteps )
            record_meanchi2 = []
            # print initial temperature ladder???
            
            # thin: The number of iterations to perform between saving the 
            # state to the internal chain.
            for iteration, s in enumerate(
                                sampler.sample(p0=pop0, iterations=cls.Itmax, 
                                thin=cls.conver_steps, storechain=True, 
                                adapt=True, swap_ratios=False)):

                if (iteration+1) % cls.conver_steps :
                    continue

                max_value, max_index = max((x, (i, j))
                                for i, row in enumerate(s[2][:])
                                for j, x in enumerate(row))

                # get_autocorr_time, returns a matrix of autocorrelation 
                # lengths for each parameter in each temperature of shape 
                # ``(Ntemps, ndim)``.
                tau = sampler.get_autocorr_time()
                mean_tau = np.mean(tau)
                # tswap_acceptance_fraction, returns an array of accepted 
                # temperature swap fractions for each temperature; 
                # shape ``(ntemps, )
                # nswap_accepted/nswap
                swap = list(sampler.tswap_acceptance_fraction)

                # acceptance_fraction, matrix of shape ``(Ntemps, Nwalkers)`` 
                # detailing the acceptance fraction for each walker.
                # nprop_accepted/nprop
                acc0 = sampler.acceptance_fraction[0,:]

                xbest = s[0][max_index[0]][max_index[1]]

                #current_meanposterior = np.mean(s[1][0][:])
                current_meanchi2 = np.mean(s[2][0][:])
                record_meanchi2.append(current_meanchi2)
                std_meanchi2 = np.std(record_meanchi2[int(index/2):])

                # Output in terminal
                if verbose:
                    print("--------- Iteration: ", iteration + 1)
                    print(" Mean tau:", round(mean_tau, 3))
                    print(" Accepted swap fraction in Temp 0: ", round(swap[0],3))
                    print(" Mean acceptance fraction Temp 0: ", round(np.mean(acc0),3))
                    #print(" Mean posterior: ", round(current_meanposterior, 6))
                    print(" Mean likelihood: ", round(current_meanchi2, 6))
                    print(" Better Chi2: ", max_index,  round(max_value,6))
                    print(" Current mean Chi2 dispersion: ", round(std_meanchi2, 6))
                    autocorr[index] = mean_tau

                """
                                Save data in hdf5 File
                """
                # Add the constant parameters to save the best solutions in the
                # current iteration. It is not applicable to chains in the mcmc
                #for k, v in sorted(PSystem.constant_params.items(), key=lambda j:j[0]):
                #    xbest = np.insert(xbest, k, v)
                    
                # shape for chains is: (temps,walkers,steps,dim)
                # It's worth saving temperatures others than 0???
                cls._save_mcmc(cls.hdf5_filename, sampler.chain[:,:,index,:], 
                            xbest, sampler.betas, autocorr, index, mean_tau, 
                            max_value, swap, max_index, iteration, current_meanchi2)

                """
                                CONVERGENCE CRITERIA
                Here you can write your favorite convergency criteria 
                """
                ##geweke()

                if verbose:
                    print(' Elapsed time: ', round((time.time() - ti)/60.,4),'min')  

                index += 1              
                if (index+1)*conver_steps > cls.Itmax:
                    print('\n--> Maximum number of iterations reached in MCMC')
                    break				

        """
        Extract best solutions from hdf5 file and write it in ascci
        """
        #cls.
        extract_best_solutions(cls.hdf5_filename)

        print("--> Reference epoch of the solutions: ", PSystem.T0JD, " [JD]")
        print('--> Iterations performed: ', iteration +1)
        print('--> Elapsed time in MCMC:', round((time.time() - ti)/60.,4), 
                    'minutes')

        return sampler
    

    def _set_hdf5(self, PSystem, hdf5_filename):

        nsteps = -(-self.Itmax // self.conver_steps)
        self.nsteps = nsteps

        with h5py.File(hdf5_filename, 'w') as newfile:
            # Empty datasets
            NCD = newfile.create_dataset

            # SHOULD INCLUDE INFO OF THE CONSTANT PARAMS 
            # INCLUDE BOUNDARIES
            
            # System name
            newfile['NAME'] = PSystem.system_name

            # shape for chains is: (temps,walkers,steps,dim)
            NCD('CHAINS', (self.ntemps, self.nwalkers, nsteps, len(PSystem.bounds)),
                        dtype='f8', compression= "lzf")
            NCD('AUTOCORR', (nsteps,), dtype='f8', compression="gzip", 
                        compression_opts=4)
            NCD('BETAS',(nsteps, self.ntemps), dtype='f8', compression="gzip", 
                        compression_opts=4)
            NCD('TAU_PROM0', (nsteps,), dtype='f8', compression="gzip", 
                        compression_opts=4)
            NCD('ACC_FRAC0', (nsteps,self.ntemps), dtype='f8', 
                        compression="gzip", compression_opts=4)
            NCD('INDEX', (1,), dtype='i8')
            NCD('ITER_LAST', (1,), dtype='i8')

            # Save the best solution per iteration
            NCD('BESTSOLS', (nsteps, PSystem.ndim,), dtype='f8',  # PSystem.NPLA*7,
                        compression="gzip", compression_opts=4)
            NCD('BESTCHI2', (nsteps,), dtype='f8', 
                        compression="gzip", compression_opts=4)
            NCD('MEANCHI2', (nsteps,), dtype='f8', 
                        compression="gzip", compression_opts=4)

            # Constants
            NCD('POP0', self.pop0.shape, dtype='f8', 
                        compression="gzip", compression_opts=4)[:] = self.pop0
            NCD('BOUNDS', np.array(PSystem.bounds).shape, 
                        dtype='f8')[:] = PSystem.bounds  
            # TODO: Include parameterized bounds!!!  
            NCD('NDIM', (1,), dtype='i8')[:] = PSystem.ndim
            NCD('NTEMPS', (1,), dtype='i8')[:] = self.ntemps
            NCD('NWALKERS', (1,), dtype='i8')[:] = self.nwalkers
            NCD('CORES', (1,), dtype='i8')[:] = self.cores
            NCD('ITMAX', (1,), dtype='i8')[:] = self.Itmax
            NCD('CONVER_STEPS', (1,), dtype='i8')[:] = self.conver_steps
            NCD('REF_EPOCH', (1,), dtype='i8')[:] = PSystem.T0JD

            # COL_NAMES are the identifiers of each dimension
            newfile['COL_NAMES'] = PSystem.params_names
            
            # Parameters of the simulated system
            NCD('NPLA', (1,), dtype='i8')[:] = PSystem.NPLA
            NCD('MSTAR', (1,), dtype='f8')[:] = PSystem.mstar
            NCD('RSTAR', (1,), dtype='f8')[:] = PSystem.rstar
        
        return


    @staticmethod
    def _save_mcmc(hdf5_filename, current_sampler_chain, xbest, sampler_betas, autocorr, 
                   index, tau_mean, max_value, swap, max_index, iteration, meanchi2):
        
        #print(' Saving...')
        ta = time.time()  # Monitor the time wasted in saving..
        with h5py.File(hdf5_filename, 'r+') as file:
            # shape for chains is: (temps,walkers,steps,dim)
            file['CHAINS'][:,:,index,:] = current_sampler_chain
            file['BETAS'][index,:] = sampler_betas
            file['AUTOCORR'][:] = autocorr
            file['INDEX'][:] = index
            file['TAU_PROM0'][index] = tau_mean #repetido con autocorr
            file['ACC_FRAC0'][index,:] = swap
            file['ITER_LAST'][:] = iteration + 1
            # Best set of parameters in the current iteration
            file['BESTSOLS'][index,:] = xbest
            # Chi2 of that best set of parameters
            file['BESTCHI2'][index] = max_value
            file['MEANCHI2'][index] = meanchi2
        print(f' Saving time: {(time.time() - ta) :.5f} sec')
        return


    @staticmethod
    def restart_mcmc(PSystem, from_hdf5_file='', Itmax=100, conver_steps=2, cores=1, 
        suffix='_rerun', restart_ladder=False): #  ntemps=None, Tmax=None,
        
        assert suffix != '', "New HDF5 file name cannot coincide with previous\
             run. Try changing -suffix- name."
        
        #TODO: new hdf5 file must coincide with hdf5_file + suffix
        
        print("\n=========== RESTARTING MCMC ===========\n")
        print('Restarting from file: ', from_hdf5_file)

        f = h5py.File(from_hdf5_file, 'r')
        index = f['INDEX'].value[0]
        nwalkers= f['NWALKERS'].value[0]

        if restart_ladder  == True:
            ladder =  f['BETAS'].value[0]
            ladder_verbose = "Restarting temperature ladder"
        if restart_ladder  == False:
            ladder =  f['BETAS'].value[index]
            ladder_verbose = "Temperature ladder continued"

        # Take the last chain state
        init_pop = f['CHAINS'].value[:,:,index,:] 
        
        f.close()

        print('Temperature ladder status: ', ladder_verbose)
        print('Nwalkers: ',nwalkers)
        print('Ntemps: ', len(ladder))
        print('Thinning: ', conver_steps)
        print('Initial population shape: ', init_pop.shape)

        sampler = MCMC.run_mcmc(PSystem,  
                                Itmax=Itmax, 
                                conver_steps=conver_steps,
                                cores=cores,
                                nwalkers=nwalkers,  
                                ntemps=None, 
                                Tmax=None, 
                                betas=ladder, 
                                pop0=init_pop, 
                                suffix=suffix)
        return sampler


    @staticmethod
    def _logp(x, psystem=None):
        return 0.0