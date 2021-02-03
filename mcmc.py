import time 
import datetime
import numpy as np
import sys
from dataclasses import dataclass
from multiprocessing import Pool
from .utils import * # Miztli: from NAU.utils import *
from .utils import writefile, intervals, _chunks, cube_to_physical, _remove_constants # Miztli: from NAU.utils import writefile, intervals

import h5py
import ptemcee as pt
from contextlib import closing

__doc__ = "MCMC"

__all__ = ["MCMC"]



# ------------- M A R K O V  C H A I N  M O N T E - C A R L O -------------

@dataclass
class MCMC:

    """Perform an MCMC using Parallel-Tempering"""

    PSystem : None
    nwalkers : int = None
    ntemps : int = None
    itmax : int = 100 # run_time ??
    conver_steps : int = 1
    opt_data : list = None
    fbest : float = 1.0
    distribution : str = ''
    p0 : list = None 
    cores : int = 1
    tmax : float = None
    betas : list = None
    file_name : str = None
    path : str = './'
    suffix : str = ''
    verbose : bool = True


    def run(self):
        """
        Run the parallel-tempering algorithm
        """

        # temperatures from betas
        if self.betas is not None:
            self.ntemps = len(self.betas)


        if self.nwalkers < 2*self.PSystem.ndim:
            sys.exit(f"Number of walkers must be >= 2*ndim, i.e., \
                nwalkers >= {2 * self.PSystem.ndim}.\n Stopped simulation!")
        
        # ==== Initial walkers population
        if self.p0 is not None:
            self.p0 = self.p0

        elif self.opt_data is not None:
            self.p0 = init_walkers(self.PSystem, distribution=self.distribution,
                                    opt_data=self.opt_data, ntemps=self.ntemps,
                                    nwalkers=self.nwalkers,fbest=self.fbest)
        else:
            sys.exit("Invalid arguments for initial population ")
        
        # Update ndim and nwalkers
        self.ntemps, self.nwalkers, _ = self.p0.shape


        # Prepare running
        ti = time.time()
        now = datetime.datetime.now()

        print("\n =========== PARALLEL-TEMPERING MCMC ===========\n")
        print("--> Starting date: ", now.strftime("%Y-%m-%d %H:%M"))
        print("--> Reference epoch of the solutions: ", self.PSystem.T0JD, " [JD]")

        """ 
                Create and set the hdf5 file for saving the output.
        """
        
        # hdf5 file name to save mcmc data
        if self.file_name is not None:
            self.hdf5_filename =  f"{self.path}{self.file_name}{self.suffix}.hdf5"   
        else:
            self.hdf5_filename = f"{self.path}{self.PSystem.system_name}{self.suffix}.hdf5"
        
        print('--> Results will be saved at: ', self.hdf5_filename, '\n')
        
        # Create an h5py file
        self._set_hdf5( self.PSystem, self.hdf5_filename)

        """
                Run the MCMC using parallel-tempering algorithm
        """
        # Default values in ptemcee, 
        # Do not change it at least you have read Vousden et al. (2016):
        nu = 100. #/nwalkers 
        t0 = 10000. #/nwalkers
        a_scale = 10

        with closing(Pool(processes=self.cores)) as pool:

            sampler = pt.Sampler(
                                nwalkers=self.nwalkers,
                                dim=self.PSystem.ndim,
                                logp=self._logp, 
                                logl=log_likelihood_func,
                                ntemps=self.ntemps,
                                betas=self.betas,
                                adaptation_lag = t0,
                                adaptation_time=nu,
                                a=a_scale, 
                                Tmax=self.tmax,
                                pool=pool,
                                loglargs=(self.PSystem,), 
                                logpkwargs={'psystem':self.PSystem}
                                )

            index = 0
            autocorr = np.empty( self.nsteps )
            record_meanlogl = []
            
            # thin: The number of iterations to perform between saving the 
            # state to the internal chain.
            for iteration, s in enumerate(
                                sampler.sample(p0=self.p0, iterations=self.itmax, 
                                thin=self.conver_steps, storechain=True, 
                                adapt=True, swap_ratios=False)):

                if (iteration+1) % self.conver_steps :
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
                current_meanlogl = np.mean(s[2][0][:])
                record_meanlogl.append(current_meanlogl)
                std_meanlogl = np.std(record_meanlogl[int(index/2):])

                # Output in terminal
                if self.verbose:
                    print("--------- Iteration: ", iteration + 1)
                    print(" Mean tau:", round(mean_tau, 3))
                    print(" Accepted swap fraction in Temp 0: ", round(swap[0],3))
                    print(" Mean acceptance fraction Temp 0: ", round(np.mean(acc0),3))
                    #print(" Mean posterior: ", round(current_meanposterior, 6))
                    print(" Mean likelihood: ", round(current_meanlogl, 6))
                    print(" Maximum likelihood: ", max_index,  round(max_value,6))
                    print(" Current mean likelihood dispersion: ", round(std_meanlogl, 6))
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
                self._save_mcmc(self.hdf5_filename, 
                                sampler.chain[:,:,index,:], 
                                xbest, 
                                sampler.betas, 
                                autocorr, 
                                index, 
                                max_value, 
                                swap, 
                                max_index, 
                                iteration, 
                                current_meanlogl)

                """
                                CONVERGENCE CRITERIA
                Here you can write your favorite convergence criteria 
                """
                ##geweke()

                if self.verbose:
                    print(' Elapsed time: ', round((time.time() - ti)/60.,4),'min')  

                index += 1              
                if (index+1)*self.conver_steps > self.itmax:
                    print('\n--> Maximum number of iterations reached in MCMC')
                    break				

        """
        Extract best solutions from hdf5 file and write it in ascci
        """
        extract_best_solutions(self.hdf5_filename)

        print("--> Reference epoch of the solutions: ", self.PSystem.T0JD, " [JD]")
        print('--> Iterations performed: ', iteration +1)
        print('--> Elapsed time in MCMC:', round((time.time() - ti)/60.,4), 
                    'minutes')

        return sampler
    


    def _set_hdf5(self, PSystem, hdf5_filename):

        nsteps = -(-self.itmax // self.conver_steps)
        self.nsteps = nsteps

        with h5py.File(hdf5_filename, 'w') as newfile:
            # Empty datasets
            NCD = newfile.create_dataset

            # System name
            newfile['NAME'] = PSystem.system_name

            # shape for chains is: (temps,walkers,steps,dim)
            NCD('CHAINS', (self.ntemps, self.nwalkers, nsteps, len(PSystem.bounds)),
                        dtype='f8', compression= "lzf")
            NCD('AUTOCORR', (nsteps,), dtype='f8', compression="gzip", 
                        compression_opts=4)
            NCD('BETAS',(nsteps, self.ntemps), dtype='f8', compression="gzip", 
                        compression_opts=4)
            ##NCD('TAU_PROM0', (nsteps,), dtype='f8', compression="gzip", 
            ##            compression_opts=4)
            NCD('ACC_FRAC0', (nsteps,self.ntemps), dtype='f8', 
                        compression="gzip", compression_opts=4)
            NCD('INDEX', (1,), dtype='i8')
            NCD('ITER_LAST', (1,), dtype='i8')

            # Save the best solution per iteration
            NCD('BESTSOLS', (nsteps, PSystem.ndim,), dtype='f8',  # PSystem.NPLA*7,
                        compression="gzip", compression_opts=4)
            NCD('BESTLOGL', (nsteps,), dtype='f8', 
                        compression="gzip", compression_opts=4)
            NCD('MEANLOGL', (nsteps,), dtype='f8', 
                        compression="gzip", compression_opts=4)

            # Constants
            NCD('p0', self.p0.shape, dtype='f8', 
                        compression="gzip", compression_opts=4)[:] = self.p0
            NCD('BOUNDS', np.array(PSystem.bounds).shape, 
                        dtype='f8')[:] = PSystem.bounds  
            NCD('BOUNDSP', np.array(PSystem._bounds_parameterized).shape, 
                        dtype='f8')[:] = PSystem._bounds_parameterized  

            NCD('NDIM', (1,), dtype='i8')[:] = PSystem.ndim
            NCD('NTEMPS', (1,), dtype='i8')[:] = self.ntemps
            NCD('NWALKERS', (1,), dtype='i8')[:] = self.nwalkers
            NCD('CORES', (1,), dtype='i8')[:] = self.cores
            NCD('ITMAX', (1,), dtype='i8')[:] = self.itmax
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
                   index, max_value, swap, max_index, iteration, meanlogl):
        
        #print(' Saving...')
        ta = time.time()  # Monitor the time wasted in saving..
        with h5py.File(hdf5_filename, 'r+') as file:
            # shape for chains is: (temps,walkers,steps,dim)
            file['CHAINS'][:,:,index,:] = current_sampler_chain
            file['BETAS'][index,:] = sampler_betas
            file['AUTOCORR'][:] = autocorr
            file['INDEX'][:] = index
            ##file['TAU_PROM0'][index] = tau_mean #repetido con autocorr
            file['ACC_FRAC0'][index,:] = swap
            file['ITER_LAST'][:] = iteration + 1
            # Best set of parameters in the current iteration
            file['BESTSOLS'][index,:] = xbest
            # Chi2 of that best set of parameters
            file['BESTLOGL'][index] = max_value
            file['MEANLOGL'][index] = meanlogl
        print(f' Saving time: {(time.time() - ta) :.5f} sec')

        return



    @classmethod
    def restart_mcmc(cls, PSystem, from_hdf5_file='', itmax=100, conver_steps=2, cores=1, 
        suffix='_rerun', restart_ladder=False): #  ntemps=None, tmax=None,
        
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

        new_mcmc = MCMC(PSystem,  
                                itmax=itmax, 
                                conver_steps=conver_steps,
                                cores=cores,
                                nwalkers=nwalkers,  
                                ntemps=None, 
                                tmax=None, 
                                betas=ladder, 
                                p0=init_pop, 
                                suffix=suffix)
        
        sampler = new_mcmc.run()
        
        return sampler



    @staticmethod
    def _logp(x, psystem=None):

        return 0.0
