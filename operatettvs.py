from scipy.optimize import differential_evolution
import multiprocessing as mp
from multiprocessing import Pool
import time 
import datetime
import numpy as np
from scipy.optimize import minimize
import h5py
import sys
from contextlib import closing
import ptemcee as pt
from dataclasses import dataclass
from utils import *
import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
from plots import Plots

@dataclass
class Optimizers:
    """Do whatever you want with the TTVs data.
    """

    # ------------------------ O P T I M I Z E R ------------------------------

    def differential_evolution_nelder_mead(self, bounds, base_name, indi=0):

        f0_before = np.inf
        original_bounds = bounds
        # Recursively reduce the search radius to avoid many local minima
        for i in range(4): 	

            """ -----Differential Evolution-----
            From doc: To improve your chances of finding a global minimum 
            use higher popsize values, with higher mutation and (dithering), 
            but lower recombination values"""
            result = differential_evolution( 
                    calculate_chi2, bounds, disp=False, seed=np.random.seed(),
                    popsize=int(50/(i+1.)), maxiter=15, polish=False, 
                    mutation=(1.2,1.9), recombination=0.5,
                    args= (self,))
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

        """ -----Nealder-Mead-----
        Refine the solution starting from the best solution found in DE"""
        res = minimize(
            calculate_chi2, list(x0), method='Nelder-Mead', 
            options={'maxiter':2000, 'maxfev':20000, 'disp':False},
            args=(self,))
        x1 = res.x
        f1 = res.fun

		# Return the best result from the two methods above
        if f1 < f0 and not False in intervals(self, x1):
            info = '{} \t {} \n'.format(str(np.round(f1,5)),
                    "  ".join([str(np.round(it,5)) for it in x1]))
            print(indi+1, '\t   ', f1)

            writefile( base_name, 'a', info, '%-15s'
                            +' %-11s'*(len(info.split())-1) + '\n')
            
            return (f1, list(x1))

        else:
            info = '{} \t {} \n'.format(str(np.round(f0,5)), 
                    "  ".join([str(np.round(it,5)) for it in x0]))
            print(indi+1, '\t   ', f0)
            writefile( base_name, 'a', info, '%-15s'
                            +' %-11s'*(len(info.split())-1) + '\n')
            
            return (f0, list(x0))


    def run_optimizers(self, cores=1, niter=1, base_name='OPT_'):

        ta = time.time()
        now = datetime.datetime.now()

        print("\n =========== OPTIMIZATION ===========\n")
        print("Starting date: ", now.strftime("%Y-%m-%d %H:%M"))
        print('Performing ', niter, ' iterations using ', cores, ' cores')

        # Output file name
        base_name= base_name + '{}'.format(self.system_name)

        writefile( base_name, 'w', '#Beggins the optimization routine\n', 
                        '%-13s'*4 + '\n')
        header = '#Chi2 ' + " ".join(["Mass{0}[Me]       Per{0}[d]       k{0} \
                                     inc{0}[deg]         h{0}        M{0}[deg]\
                                     Ome{0}[deg]  ".format(i+1) for i in 
                                     range(self.NPLA)]) 
        writefile(base_name, "a", header, '%-15s'
                        +' %-11s'*(len(header.split())-1) + '\n')

        print('Results will be saved at: ', base_name)

        print('- - - - - - - - - - - - - - - -')
        print('Iteration    chi2_value'        )
        print('- - - - - - - - - - - - - - - -')
        ta = time.time()
        
        # Run in parallel
        with warnings.catch_warnings(record=True) as w: 
            warnings.simplefilter("always")

            pool = mp.Pool(processes=cores)
            results = [pool.apply_async(self.differential_evolution_nelder_mead,
                        args=(self.bounds, base_name,i)) for i in range(niter)]
            output = [p.get() for p in results]		
        print('Time elapsed in optimization: ', 
              (time.time() - ta)/60., 'minutes')

        return output        












# ------------- M A R K O V  C H A I N  M O N T E - C A R L O -------------

@dataclass
class MCMC:
    """Make an MCMC using Parallel-Tempering"""

    def run_mcmc(self, Itmax=100, conver_steps=2, cores=1, suffix='', 
                     nwalkers=None, ntemps=None, Tmax=None, betas=None, 
                     bounds=None, pop0=None):

        if ntemps is not None:
            self.ntemps = ntemps
        if betas is not None:
            self.ntemps = len(betas)

        self.nwalkers = nwalkers
        self.Itmax = Itmax
        self.conver_steps = conver_steps
        self.cores = cores
        self.bounds = bounds #Tal vez bounds se puede mitir porque self.bounds
        self.betas = betas
        self.Tmax = Tmax
        self.pop0 = pop0

        if self.nwalkers < 2*self.NPLA*7:
            sys.exit("Number of walkers must be >= 2*ndim, i.e., \
                nwalkers = {}.\n Stopped simulation!".format(2*self.NPLA*7))

        ti = time.time()
        now = datetime.datetime.now()
        print("\n =========== PARALLEL-TEMPERING MCMC ===========\n")
        print("--> Starting date: ", now.strftime("%Y-%m-%d %H:%M"))

        """ 
                Create and set the hdf5 file for saving the output.
        """
        # hdf5 file name to save mcmc data
        self.hdf5_filename = "{}{}.hdf5".format(self.system_name, suffix) 
        print('--> Results will be saved at: ', self.hdf5_filename, '\n')
        
        # Create an h5py file
        self.set_hdf5(self.hdf5_filename)

        """
                Run the MCMC usin parallel-tempering algorithm
        """
        # Default values in ptemcee, 
        # Do not change it at least you have read Vousden et al. (2016):
        nu = 100. #100./self.nwalkers 
        t0 = 1000. #1000./self.nwalkers
        a_scale= 2.0

        stopflag = 0 # Flag to stop MCMC when convergence is reached.
        with closing(Pool(processes=self.cores)) as pool:

            sampler = pt.Sampler(
                    nwalkers=self.nwalkers, dim=self.NPLA*7, logp=logp, 
                    logl=calculate_chi2, ntemps=self.ntemps, betas=self.betas,
                    adaptation_lag = t0, adaptation_time=nu, a=a_scale, 
                    Tmax=self.Tmax, pool=pool, loglargs=(self,), 
                    loglkwargs={'flag':'MCMC'})

            index = 0
            autocorr = np.empty( self.nsteps )
            old_tau = np.inf
            # thin: The number of iterations to perform between saving the 
            # state to the internal chain.
            for iteration, s in enumerate(
                                sampler.sample(p0=pop0, iterations=self.Itmax, 
                                thin=self.conver_steps, storechain=True, 
                                adapt=True, swap_ratios=False )):
                #print("Starts it", (iteration+1) % self.conver_steps)
                if (iteration+1) % self.conver_steps :
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
                # print 'accep swap:', swap
                # print 'betas:', sampler.betas
                # print '<posterior>: ', np.mean(s[1][0][:])
                print(' <likelihood>: ', np.mean(s[2][0][:]))
                print(' better posterior:', max_index,  max_value )
                autocorr[index] = np.mean(tau)

                """
                                Save data in hdf5 File
                """
                self.save_mcmc(sampler, self.hdf5_filename, autocorr, 
                                index, tau, max_value, swap, max_index, 
                                iteration, s)
                
                """
                                Make figure of TTVs
                """
                mejor = [(max_value, list(s[0][max_index[0]][max_index[1]]) )]
                
                Plots.plot_TTVs(self, mejor )  
                plt.savefig("TTVs_MCMC.png")
                plt.close()

                chunk = int((iteration + 1)/self.conver_steps)  
                Plots.plot_hist(self, sampler.chain[0,:,int(index/4):chunk,:])  
                plt.savefig("Hist_MCMC.png")
                plt.close()

                """
                                CONVERGENCE CRITERIA
                Here you can write your favorite convergency criteria 
                xxxxxxxxxx
                """

                old_tau = tau  
                index += 1
                print(' Elapsed time: ', round((time.time() - ti)/60.,4), 'min')                
                if (index+1)*conver_steps > self.Itmax:
                    print('\n\n--> Maximum number of iterations \
                                    reached in MCMC\n')
                    break				

            #pool.terminate()

        """
        Extract best solutions from hdf5 file and write it in ascci
        """
        self.extract_best_solutions(self.hdf5_filename)
        
        print('--> Iterations performed: ', iteration +1)
        print('--> Time elapsed in MCMC:', round((time.time() - ti)/60.,4), 
                    'minutes')

        return sampler
    

    def set_hdf5(self, hdf5_filename):
        nsteps = -(-self.Itmax // self.conver_steps)
        self.nsteps = nsteps

        with h5py.File(hdf5_filename, 'w') as newfile:
            # Empty datasets
            NCD = newfile.create_dataset
            NCD('CHAINS', (self.ntemps, self.nwalkers, nsteps, self.NPLA*7), 
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
            NCD('BESTSOLS', (nsteps, self.NPLA*7,), dtype='f8', 
                        compression="gzip", compression_opts=4)
            NCD('BESTCHI2', (nsteps,), dtype='f8', 
                        compression="gzip", compression_opts=4)
            NCD('MEANCHI2', (nsteps,), dtype='f8', 
                        compression="gzip", compression_opts=4)

            # Constants
            NCD('POP0', self.pop0.shape, dtype='f8', 
                        compression="gzip", compression_opts=4)[:] = self.pop0
            NCD('BOUNDS', np.array(self.bounds).shape, 
                        dtype='f8')[:] = self.bounds
            NCD('NTEMPS', (1,), dtype='i8')[:] = self.ntemps
            NCD('NWALKERS', (1,), dtype='i8')[:] = self.nwalkers
            NCD('CORES', (1,), dtype='i8')[:] = self.cores
            NCD('ITMAX', (1,), dtype='i8')[:] = self.Itmax
            NCD('CONVER_STEPS', (1,), dtype='i8')[:] = self.conver_steps

            # Parameters of the simulated system
            NCD('NPLA', (1,), dtype='i8')[:] = self.NPLA
            NCD('MSTAR', (1,), dtype='f8')[:] = self.mstar
            NCD('RSTAR', (1,), dtype='f8')[:] = self.rstar
        
        return


    @staticmethod
    def save_mcmc(
            sampler, hdf5_filename, autocorr, index, tau, max_value, 
            swap, max_index, iteration, s):
        Tg = time.time()
        print(' Writing...')
        with h5py.File(hdf5_filename, 'r+') as newfile:
            # shape for chains is: (temps,walkers,steps,dim)
            newfile['CHAINS'][:,:,index,:] = sampler.chain[:,:,index,:]
            newfile['AUTOCORR'][:] = autocorr
            newfile['INDEX'][:] = index
            newfile['BETAS'][index,:] = sampler.betas
            newfile['TAU_PROM0'][index] = np.mean(tau)
            newfile['ACC_FRAC0'][index,:] = swap
            newfile['ITER_LAST'][:] = iteration + 1
            # Best set of parameters in the current iteration
            newfile['BESTSOLS'][index,:] = s[0][max_index[0]][
                                                max_index[1]]
            # Chi2 of that best set of parameters
            newfile['BESTCHI2'][index] = max_value
            newfile['MEANCHI2'][index] = np.mean(s[2][0][:])

        print(' Saved!')
        print(' Time elapsed in saving: ', round(time.time()-Tg, 4), ' sec')

        return


    @staticmethod
    def extract_best_solutions(hdf5_filename):
        f = h5py.File(hdf5_filename, 'r')
        mstar = f['MSTAR'][()][0]
        rstar = f['RSTAR'][()][0]
        NPLA = f['NPLA'][()][0]
        best= f['BESTSOLS'][()]  #.value[:index+1]
        log1_chi2 = f['BESTCHI2'][()]  #.value[:index+1]
        f.close()

        # Sort solutions by chi2: from better to worst
        tupla = zip(log1_chi2,best)
        tupla_reducida = list(dict(tupla).items())
        sorted_by_chi2 = sorted(tupla_reducida, key=lambda tup: tup[0])[::-1] 

        best_file = '{}.best'.format(hdf5_filename.split('.')[0])
        head = "#Mstar[Msun]      Rstar[Rsun]     Nplanets"
        writefile(best_file, 'w', head, '%-10s '*3 +'\n')
        head = "{}             {}           {}\n".format(mstar, rstar, NPLA)
        writefile(best_file, 'a', head, '%-10s '*3 +'\n')

        head = "#Idx    -Chi2  " + " ".join([
            "m{0}[Mearth]  Per{0}[d]  k{0}  inc{0}[deg]  h{0}[deg]  M{0}[deg]\
            Ome{0}[deg] ".format(i+1) for i in range(NPLA)]) + '\n'

        writefile(best_file, 'a', head,  '%-8s '+'%-16s'+' %-11s'*(
                                                len(head.split())-2) + '\n')
        for idx, s in enumerate(sorted_by_chi2):
            texto = str(idx)+ ' ' + str(round(s[0],5))+ ' ' + \
                        " ".join(str(round(i,5)) for i in s[1]) 
            writefile(best_file, 'a', texto,  '%-8s '+'%-16s' + \
                        ' %-11s'*(len(texto.split())-2) + '\n')
        print('--> Best solutions from the MCMC will be written at: ',best_file)

        return



