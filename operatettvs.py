from dataclasses import dataclass
from multiprocessing import Pool
from contextlib import closing
from scipy.optimize import differential_evolution
from scipy.optimize import minimize
from utils import *
import multiprocessing as mp
import matplotlib.pyplot as plt
import time 
import datetime
import numpy as np
import copy
import h5py
import sys
import ptemcee as pt
import warnings
warnings.filterwarnings("ignore")


# ------------------------ O P T I M I Z E R ------------------------------

@dataclass
class Optimizers:
    """Fit the TTVs using differential evolution + Nelder Mead optimization 
        routines
    """

    @staticmethod
    def _differential_evolution_nelder_mead(PSystem, base_name, indi=0):
        
        f0_before = np.inf
        original_bounds = copy.deepcopy(PSystem.bounds)
        bounds = copy.deepcopy(PSystem.bounds)

        """ 
                        -----Differential Evolution----- 
        """
        # Recursively reduce the search radius to avoid many local minima
        for i in range(4): 	
            # From doc: To improve your chances of finding a global minimum 
            # use higher popsize values, with higher mutation and (dithering), 
            # but lower recombination values
            
            
            result = differential_evolution( 
                    calculate_chi2, bounds, disp=False, seed=np.random.seed(),
                    popsize=int(50/(i+1.)), maxiter=15, polish=False, 
                    mutation=(1.2,1.9), recombination=0.5,
                    args= (PSystem,))
                    # popsize=30, maxiter=100, polish=True, mutation=(1.2,1.9),
                    # recombination=0.7) #50,30
            x0 = result.x
            f0 = result.fun

            if f0 < f0_before:
                # Reduce the boundaries by a factor 4 around the previous
                # best solution.
                
                length = [(bounds[j][1] - bounds[j][0])*.33 
                            for j in range(len(bounds))]
                
                bounds = [[max(x0[k]-length[k], original_bounds[k][0]), 
                           min(x0[k]+length[k], original_bounds[k][1])] 
                            for k in range(len(length))]
                
                f0_before = f0
            else:
                break

        """ 
                            -----Nealder-Mead-----
        Refine the solution starting from the best solution found by DE
        """
        res = minimize(
            calculate_chi2, list(x0), method='Nelder-Mead', 
            options={'maxiter':2000, 'maxfev':20000, 'disp':False},
            args=(PSystem,))
        x1 = res.x
        f1 = res.fun

		# Return the best result from the two methods above
        if f1 < f0 and not False in intervals(PSystem.bounds, x1):
            # Insert again constan values
            for k, v in sorted(PSystem.constant_params.items(), key=lambda j:j[0]):
                x1 = np.insert(x1, k, v)

            info = '{} \t {} \n'.format(str(np.round(f1,5)),
                    "  ".join([str(np.round(it,5)) for it in x1]))
            print(indi+1, '\t   ', f1)

            writefile( base_name, 'a', info, '%-15s'
                            +' %-11s'*(len(info.split())-1) + '\n')
            
            return ([f1] + list(x1))

        else:
            info = '{} \t {} \n'.format(str(np.round(f0,5)), 
                    "  ".join([str(np.round(it,5)) for it in x0]))
            print(indi+1, '\t   ', f0)
            writefile( base_name, 'a', info, '%-15s'
                            +' %-11s'*(len(info.split())-1) + '\n')
            
            return ([f0] + list(x0))


    @classmethod
    def run_optimizers(cls, PSystem, Ftime='Auto', cores=1, niter=1, 
                        base_name='_OPT'):

        ta = time.time()
        now = datetime.datetime.now()

        print("\n =========== OPTIMIZATION ===========\n")
        print("--> Starting date: ", now.strftime("%Y-%m-%d %H:%M"))
        print("--> Performing ", niter, " iterations using ", cores, " cores")

        # Output file name
        base_name = '{}'.format(PSystem.system_name) + base_name

        writefile( base_name, 'w', '#Results of the optimizers\n', 
                        '%-13s'*4 + '\n')
        header = '#Chi2 ' + " ".join(["Mass{0}[Me]       Per{0}[d]       ecc{0} \
                                     inc{0}[deg]         arg{0}        M{0}[deg]\
                                     Ome{0}[deg]  ".format(i+1) for i in 
                                     range(PSystem.NPLA)]) 
        writefile(base_name, "a", header, '%-15s'
                        +' %-11s'*(len(header.split())-1) + '\n')

        print("--> Results will be saved at: ", base_name)

        print('- - - - - - - - - - - - - - - -')
        print('Iteration    chi2_value'        )
        print('- - - - - - - - - - - - - - - -')
        ta = time.time()
        
        # Run in parallel
        with warnings.catch_warnings(record=True) as w: 
            warnings.simplefilter("always")

            pool = mp.Pool(processes=cores)
            results = [pool.apply_async(cls._differential_evolution_nelder_mead,
                        args=(PSystem, base_name, i)) for i in range(niter)]
            output = [p.get() for p in results]		
        print('Time elapsed in optimization: ', 
              (time.time() - ta)/60., 'minutes')

        return output        



# ------------- M A R K O V  C H A I N  M O N T E - C A R L O -------------

@dataclass
class MCMC:

    """Perform an MCMC using Parallel-Tempering"""

    @classmethod
    def run_mcmc(cls, PSystem, Ftime='Auto', Itmax=100, conver_steps=2, cores=1,
                 nwalkers=None, ntemps=None, Tmax=None, betas=None, pop0=None, 
                 suffix=''):

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

        if cls.nwalkers < 2*PSystem.NPLA*7:
            sys.exit("Number of walkers must be >= 2*ndim, i.e., \
                nwalkers = {}.\n Stopped simulation!".format(2*PSystem.NPLA*7))

        ti = time.time()
        now = datetime.datetime.now()
        print("\n =========== PARALLEL-TEMPERING MCMC ===========\n")
        print("--> Starting date: ", now.strftime("%Y-%m-%d %H:%M"))

        """ 
                Create and set the hdf5 file for saving the output.
        """
        # hdf5 file name to save mcmc data
        cls.hdf5_filename = "{}{}.hdf5".format(PSystem.system_name, suffix) 
        print('--> Results will be saved at: ', cls.hdf5_filename, '\n')
        
        # Create an h5py file
        cls._set_hdf5(cls, PSystem, cls.hdf5_filename)

        """
                Run the MCMC usin parallel-tempering algorithm
        """
        # Default values in ptemcee, 
        # Do not change it at least you have read Vousden et al. (2016):
        nu = 100. #100./PSystem.nwalkers 
        t0 = 1000. #1000./PSystem.nwalkers
        a_scale= 2.0

        with closing(Pool(processes=cls.cores)) as pool:

            sampler = pt.Sampler(
                    nwalkers=cls.nwalkers, dim=len(PSystem.bounds),
                    logp=logp, 
                    logl=calculate_chi2, ntemps=cls.ntemps, betas=cls.betas,
                    adaptation_lag = t0, adaptation_time=nu, a=a_scale, 
                    Tmax=cls.Tmax, pool=pool, loglargs=(PSystem,), 
                    loglkwargs={'flag':'MCMC'})

            index = 0
            autocorr = np.empty( cls.nsteps )
            
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

                # get_autocorr_time:  Returns a matrix of autocorrelation 
                # lengths for each parameter in each temperature of shape 
                # ``(Ntemps, Ndim)``.
                tau = sampler.get_autocorr_time()#[0]
                swap = list(sampler.tswap_acceptance_fraction)

                print("--------- Iteration: ", iteration +1)
                print(" <tau> :", np.mean(tau))
                print(" acceptance fraction Temp 0: ", 
                        round(np.mean(sampler.acceptance_fraction[0,:]),5) )
                print(' <likelihood>: ', np.mean(s[2][0][:]))
                print(' better posterior:', max_index,  max_value )
                autocorr[index] = np.mean(tau)

                """
                                Save data in hdf5 File
                """
                xbest = s[0][max_index[0]][max_index[1]]

                # Add the constant parameters
                for k, v in sorted(PSystem.constant_params.items(), key=lambda j:j[0]):
                    xbest = np.insert(xbest, k, v)
                    
                # shape for chains is: (temps,walkers,steps,dim)
                cls._save_mcmc(cls.hdf5_filename, sampler.chain[:,:,index,:], 
                               xbest, sampler.betas, autocorr, index, np.mean(tau), 
                               max_value, swap, max_index, iteration, s)

                """
                                CONVERGENCE CRITERIA
                Here you can write your favorite convergency criteria 
                xxxxxxxxxx
                """

                index += 1
                print(' Elapsed time: ', round((time.time() - ti)/60.,4),'min')                
                if (index+1)*conver_steps > cls.Itmax:
                    print('\n--> Maximum number of iterations reached in MCMC')
                    break				

            #pool.terminate()

        """
        Extract best solutions from hdf5 file and write it in ascci
        """
        cls.extract_best_solutions(cls.hdf5_filename)
        
        print('--> Iterations performed: ', iteration +1)
        print('--> Time elapsed in MCMC:', round((time.time() - ti)/60.,4), 
                    'minutes')

        return sampler
    

    def _set_hdf5(self, PSystem, hdf5_filename):

        nsteps = -(-self.Itmax // self.conver_steps)
        self.nsteps = nsteps

        with h5py.File(hdf5_filename, 'w') as newfile:
            # Empty datasets
            NCD = newfile.create_dataset
            # COL_NAMES are the identifiers of each dimension
            newfile['COL_NAMES'] = PSystem.params_names
            # shape for chains is: (temps,walkers,steps,dim)
            NCD('CHAINS', (self.ntemps, self.nwalkers, nsteps, len(PSystem.bounds)), #!!! 
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
            NCD('BESTSOLS', (nsteps, PSystem.NPLA*7,), dtype='f8', #!!!
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
            NCD('NTEMPS', (1,), dtype='i8')[:] = self.ntemps
            NCD('NWALKERS', (1,), dtype='i8')[:] = self.nwalkers
            NCD('CORES', (1,), dtype='i8')[:] = self.cores
            NCD('ITMAX', (1,), dtype='i8')[:] = self.Itmax
            NCD('CONVER_STEPS', (1,), dtype='i8')[:] = self.conver_steps

            # Parameters of the simulated system
            NCD('NPLA', (1,), dtype='i8')[:] = PSystem.NPLA
            NCD('MSTAR', (1,), dtype='f8')[:] = PSystem.mstar
            NCD('RSTAR', (1,), dtype='f8')[:] = PSystem.rstar
        
        return


    @staticmethod
    def _save_mcmc(hdf5_filename, sampler_chain, xbest, sampler_betas, autocorr, 
                   index, tau_mean, max_value, swap, max_index, iteration, s):
        
        Tg = time.time()
        print(' Writing...')
        with h5py.File(hdf5_filename, 'r+') as newfile:
            # shape for chains is: (temps,walkers,steps,dim)
            newfile['CHAINS'][:,:,index,:] = sampler_chain
            newfile['BETAS'][index,:] = sampler_betas
            newfile['AUTOCORR'][:] = autocorr
            newfile['INDEX'][:] = index
            newfile['TAU_PROM0'][index] = tau_mean
            newfile['ACC_FRAC0'][index,:] = swap
            newfile['ITER_LAST'][:] = iteration + 1
            # Best set of parameters in the current iteration
            newfile['BESTSOLS'][index,:] = xbest
            # Chi2 of that best set of parameters
            newfile['BESTCHI2'][index] = max_value
            newfile['MEANCHI2'][index] = np.mean(s[2][0][:])
        print(' Saved!')
        print(' Elapsed time in saving: ', round(time.time()-Tg, 4), ' sec')

        return


    @staticmethod
    def extract_best_solutions(hdf5_filename):
        f = h5py.File(hdf5_filename, 'r')
        mstar = f['MSTAR'][()][0]
        rstar = f['RSTAR'][()][0]
        NPLA = f['NPLA'][()][0]
        best= f['BESTSOLS'][()] 
        log1_chi2 = f['BESTCHI2'][()] 
        f.close()

        # Sort solutions by chi2: from better to worst
        tupla = zip(log1_chi2,best)
        tupla_reducida = list(dict(tupla).items())
        sorted_by_chi2 = sorted(tupla_reducida, key=lambda tup: tup[0])[::-1] 

        best_file = '{}.best'.format(hdf5_filename.split('.')[0])
        head = "#Mstar[Msun]      Rstar[Rsun]     Nplanets"
        writefile(best_file, 'w', head, '%-10s '*3 +'\n')
        head = "#{}            {}           {}\n".format(mstar, rstar, NPLA)
        writefile(best_file, 'a', head, '%-10s '*3 +'\n')

        
        head = "#-Chi2  " + " ".join([
            "m{0}[Mearth]  Per{0}[d]  ecc{0}  inc{0}[deg]  arg{0}[deg]  M{0}[deg]\
            Ome{0}[deg] ".format(i+1) for i in range(NPLA)]) + '\n'

        writefile(best_file, 'a', head, '%-16s'+' %-11s'*(
                                                len(head.split())-1) + '\n')
        
        for _, s in enumerate(sorted_by_chi2):
            texto =  ' ' + str(round(s[0],5))+ ' ' + \
                        " ".join(str(round(i,5)) for i in s[1]) 
            writefile(best_file, 'a', texto,  '%-16s' + \
                        ' %-11s'*(len(texto.split())-1) + '\n')
        print('--> Best solutions from the MCMC will be written at: ',best_file)

        return