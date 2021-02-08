import time 
import datetime
import numpy as np
from dataclasses import dataclass
from multiprocessing import Pool
from .utils import * # Miztli: from NAU.utils import *
from .utils import writefile, intervals, _chunks, cube_to_physical, _remove_constants # Miztli: from NAU.utils import writefile, intervals
import h5py
import ptemcee as pt
from contextlib import closing


__doc__ = "A module to perform MCMC runs using the Parallel-tempering algorithm"

__all__ = ["MCMC"]


@dataclass
class MCMC:
    """Perform an MCMC run using Parallel-Tempering algorithm

    Parameters
    ----------
    PSystem : 
        The Planetary System object
    p0 : array, optional
        Initial population of walkers. Shape must be (ntemps, nwalkers, ndim),
        by default None, in which case ntemps, nwalkers, opt_data, fbest  and 
        distribution must be specified. If p0 is given, then ignore ntemps, 
        nwalkers, opt_data, fbest and distribution parameters.
    ntemps : int, optional
        Number of temperatures for the parallel-tempering MCMC. If p0 is not
        None, ntemps is taken from p0 shape[0].
    nwalkers : int, optional
        Number of walkers per temperature. If p0 is not None, nwalkers is taken
        from p0 shape[1].
    opt_data : dict or array, optional
        Results from the optimizers that will be used to create p0.
        -If dict, it have to be the dictionary comming from the optimizers with 
        keys 'chi2', 'cube', 'physical'.
        -If array, it have to be an array created from the file '*_cube.opt',
        for example, using numpy.genfromtxt().
    fbest : float, optional
        A fraction between 0 and 1 to especify the fraction of best solutions
        from opt_data (if given) to create p0, by default 1.
    distribution : str, optional
        A distribution name to create p0 if ntemps, nwalkers, opt_data and 
        fbest are given. The current supported distributions are: 'uniform', 
        'gaussian', 'picked', 'ladder'. See utils.init_walkers for details
        about these distributions.
    itmax : int
        Number of maximum iterations performed in the MCMC, by default 100.
    intra_steps : int
        Number of internal steps for saving the state of the chains, by
        default 1.
    tmax : float, optional
        Maximum temperature value in the temperature ladder. By default it is
        calculated from ptemcee.
    betas : list
        A list of inverse temperatures for the temperature ladder. By default,
        it is calculated from ptemcee.
    cores : int
        Number of cores to run in parallel, by default 1.
    file_name : str
        The file name of the output .hdf5 file where the main features of the
        MCMC run are saved. By default the file_name corresponds to the name
        of the Planetary System.
    path : str
        A directory path to save the output file, by default './'.
    suffix : str
        A suffix for the output file name, by default ''.
    verbose : bool
        A flag to print a summary of the current status of the run at each 
        output especified by intra_steps, by default True.
    """


    PSystem : None
    p0 : list = None 
    ntemps : int = None
    nwalkers : int = None
    opt_data : list = None
    fbest : float = 1.0
    distribution : str = ''
    itmax : int = 100 # run_time ??
    intra_steps : int = 1
    tmax : float = None
    betas : list = None
    cores : int = 1
    file_name : str = None
    path : str = './'
    suffix : str = ''
    verbose : bool = True


    def run(self):
        """
        Run the parallel-tempering algorithm
        """

        # PREPARE FOR RUNNING

        # Define initial walkers population
        if self.p0 is not None:
            pass

        elif type(None) not in ( type(self.opt_data), type(self.ntemps), 
                                        type(self.nwalkers), type(self.fbest)):
            self.p0 = init_walkers(self.PSystem,distribution=self.distribution,
                                    opt_data=self.opt_data, ntemps=self.ntemps,
                                    nwalkers=self.nwalkers,fbest=self.fbest)
        else:
            raise NameError("No 'p0' have been encountered. Provide 'p0' or " +
                            "define: 'opt_data', 'fbest', 'ntemps', " +
                            "'nwalkers', and 'distribution' to initialize " +
                            "walkers from optimizers.")
        
        # Update ndim and nwalkers from p0 above
        self.ntemps, self.nwalkers, ndim_tmp = self.p0.shape

        # Check for consistency in input parameters
        if self.nwalkers < 2 * self.PSystem.ndim:
            raise RuntimeError(f"Number of walkers must be >= 2*ndim, i.e., " +
                f"nwalkers have to be >= {2 * self.PSystem.ndim}.")

        if ndim_tmp != self.PSystem.ndim:
            raise RuntimeError(f"Number of dimensions in 'PSystem' " +
            f"({self.PSystem.ndim}) differs from that in 'p0' ({ndim_tmp}).")
        
        # temperatures from betas
        if self.betas is not None:
            if len(self.betas) == self.ntemps:
                pass
            else:
                raise RuntimeError(f"Number of 'betas' ({self.betas}) differs"+
                f" from number of temperatures in 'p0' ({self.ntemps})")

        # hdf5 file name to save mcmc data
        if self.file_name is not None:
            self.hdf5_filename =  f"{self.path}{self.file_name}{self.suffix}.hdf5"   
        else:
            self.hdf5_filename = f"{self.path}{self.PSystem.system_name}{self.suffix}.hdf5"


        # Time it
        ti = time.time()
        now = datetime.datetime.now()
        print("\n =========== PARALLEL-TEMPERING MCMC ===========\n")
        print("--> Starting date: ", now.strftime("%Y-%m-%d %H:%M"))
        print("--> Reference epoch of the solutions: ", self.PSystem.T0JD, " [JD]")
        print('--> Results will be saved at: ', self.hdf5_filename)
        print("--> MCMC parameters:")
        print(f"      -ntemps: {self.ntemps}")
        print(f"      -nwalkers: {self.nwalkers}")
        print(f"      -itmax: {self.itmax}")
        print(f"      -intra_steps: {self.intra_steps}")
        print()

        # Create an h5py file
        self._set_hdf5( self.PSystem, self.hdf5_filename)
        
        # Default values in ptemcee, 
        # Do not change it at least you have read Vousden et al. (2016):
        nu = 100. #/nwalkers 
        t0 = 10000. #/nwalkers
        a_scale = 10

        # RUN
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
                                thin=self.intra_steps, storechain=True, 
                                adapt=True, swap_ratios=False)):

                if (iteration+1) % self.intra_steps :
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

                
                # Save data in hdf5 File
                    
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
                if (index+1)*self.intra_steps > self.itmax:
                    print('\n--> Maximum number of iterations reached in MCMC')
                    break				

        
        # Extract best solutions from hdf5 file and write it in ascci
        extract_best_solutions(self.hdf5_filename, write_file=True)

        print("--> Reference epoch of the solutions: ", self.PSystem.T0JD, " [JD]")
        print('--> Iterations performed: ', iteration +1)
        print('--> Elapsed time in MCMC:', round((time.time() - ti)/60.,4), 
                    'minutes')

        return sampler
    


    def _set_hdf5(self, PSystem, hdf5_filename):
        """Generates an hdf5 file and set fields

        Parameters
        ----------
        PSystem : 
            The Planetary System object
        hdf5_filename : str
            The name of the .hdf file
        """

        nsteps = -(-self.itmax // self.intra_steps)
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
            NCD('INTRA_STEPS', (1,), dtype='i8')[:] = self.intra_steps
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
        """A help function to save the current state of the MCMC run"""
        
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
    def restart_mcmc(cls, PSystem, hdf5_file, in_path='./', out_path='./', 
        itmax=100, intra_steps=2, cores=1, suffix='_2', restart_ladder=False):
        """A function to restart a MCMC simulation from previous hdf5 file.

        Parameters
        ----------
        PSystem : 
            The Planetary System object.
        hdf5_file : str
            The name of a hdf5 file to restart the MCMC run. If this file is
            in a different directory than the working directory, provide the
            route through 'in_path'. By default, the new file will be saved 
            at the same directory, at least you specify other in 'out_path'.
            For consistency, the output file name will be equal to hdf5_file
            plus suffix.
        in_path : str, optional
            The path where the input file is, by default './'.
        out_path
            The path where the output file will be saved, by default './'.
        itmax : int, optional
            Number of maximum iterations performed in the MCMC, by default 100.
        intra_steps : int, optional
            Number of internal steps for saving the state of the chains, by
            default 2.
        cores : int, optional
            Number of cores to run in parallel, by default 1.
        suffix : str, optional
            A suffix for the output file name, by default '_2'.
        restart_ladder : bool, optional
            A flag to restart the temperature ladder (True) or keep the last
            state of the ladder of previous run (False), by default False.

        Returns
        -------
        dict
            The sampler object in dictionary
        """
        from os import path

        # Verify output directory exists
        if path.exists(f'{out_path}'):
            if out_path.endswith('/'):
                pass
            else:
                out_path += '/'
        else:
            raise RuntimeError(f"directory {out_path} does not exists")

        # Verify entry data are correct
        if hdf5_file != '':
            if in_path.endswith('/'):
                file_in = in_path + hdf5_file
            else:
                file_in = in_path + '/' + hdf5_file
        else:
            raise RuntimeError("To restart, provide 'PSystem' and 'hdf5_file'")

        assert suffix != '', ("New HDF5 file name cannot coincide with previous"+
            " run. Try changing 'suffix' name.")


        # Read previous run
        f = h5py.File(file_in, 'r')
        index = f['INDEX'].value[0]

        if restart_ladder  == True:
            ladder =  f['BETAS'].value[0]
            ladder_verbose = "Restarting temperature ladder"
        if restart_ladder  == False:
            ladder =  f['BETAS'].value[index]
            ladder_verbose = "Temperature ladder continued"

        # Take the last chain state
        init_pop = f['CHAINS'].value[:,:,index,:] 
        
        f.close()

        out_file = hdf5_file.split('.')[-2]

        print("\n =========== RESTARTING MCMC ===========\n")
        print('--> Restarting from file: ', file_in)
        print('--> Temperature ladder status: ', ladder_verbose)
        print()

        new_mcmc = MCMC(PSystem,  
                                itmax=itmax, 
                                intra_steps=intra_steps,
                                cores=cores,
                                betas=ladder, 
                                p0=init_pop, 
                                file_name=out_file,
                                path =out_path,
                                suffix=suffix)
        
        sampler = new_mcmc.run()
        
        return sampler


    @staticmethod
    def _logp(x, psystem=None):
        """The log prior function"""

        return 0.0
