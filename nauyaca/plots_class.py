import seaborn as sns
from sklearn.linear_model import LinearRegression
from .utils import *
import matplotlib
import matplotlib.pyplot as plt
from collections.abc import Iterable
from mpl_toolkits.axes_grid1 import make_axes_locatable
import corner 
import h5py
import pandas as pd
import numpy as np
from dataclasses import dataclass

from .constants import colors, units_latex, labels

__all__ = ["Plots_c"]


# TODO: Include functions to visualize Optimizer results

@dataclass
class Plots_c:


    PSystem : None
    hdf5_file : str = None
    temperature : int = 0
    burnin : float = 0.0  
    sns_context : str = 'notebook'
    sns_style : str = 'darkgrid'
    size : tuple = (10,10)  # Size debe ser kwarg de cada plot con predefinidos
    colors = colors



    def plot_TTVs(self, flat_params=None, mode='None', nsols=1, show_obs=True, residuals=True):

        """Plot the observed TTVs signals and the transit solutions from flat_params
        
        Arguments:
            flat_params {Array} -- A flat array containing mass, period, eccentricity,
                inclination, argument, mean anomaly, ascending node for all planets.
                It could be included more than one solution.
        """

        # Be aware of the constant parameters
        mins = 1440.
        sns.set(context=self.sns_context, style=self.sns_style)
        nplots = len(self.PSystem.TTVs) #self.PSystem.NPLA #



        
        #sns.despine()
        
        # =====================================================================
        # PREPARING DATA
        # flat_params have to:
        #   be a list of lists!
        #   be in physical values with all the dimensions

        try:
            flat_params = list(flat_params)
        except:
            pass

        if flat_params != None:
            # Is a list or array
            if isinstance(flat_params[0], float):
                flat_params = list(flat_params)  # convert to list
                flat_params = self._check_dimensions(flat_params)
                flat_params = [flat_params]   # convert to list of list

            # Items are iterables
            elif isinstance(flat_params[0], Iterable):
                flat_params = [list(fp) for fp in flat_params ]
                flat_params = [self._check_dimensions(fp) for fp in flat_params]
                
            else: 
                raise ValueError("Invalid items in flat_params. Items must be planet parameters or list of solutions")
            """
            if (flat_params != None) and (isinstance(flat_params[0], list)):
                # flat_params is a list of lists
                print("E1")
                flat_params = []
                
                # Verify items are lists
                fp_tmp = []
                for fp in flat_params:
                    if isinstance(fp,list):
                        fp_tmp.append(self._check_dimensions(self.PSystem, fp))
                    else:
                        raise ValueError("Items in flat_params must be lists")
                flat_params = fp_tmp
                #pass

            elif (flat_params != None) and (isinstance(flat_params[0], list)==False):
                # flat_params is a unique list
                print("E2")

                flat_params = self._check_dimensions(self.PSystem, flat_params)
                flat_params = [flat_params]
            """
        elif self.hdf5_file != None:
            # Try to build flat_params from hdf5 and the attributes of this function
            print("E3")
            if mode.lower() == 'random':
                #FIXME: There is a bug when random mode is active:
                # Sometimes a random solution without enough data is selected,
                # Producing KeyError
                print("random")

                r = get_mcmc_results(self.hdf5_file, keywords=['INDEX','CHAINS'])
                index = int(r['INDEX'][0])
                burnin_ = int(self.burnin*(index+1))
                chains  = r['CHAINS'][0,:,burnin_:index+1,:]
                wk, it, _ = chains.shape 
                del r 

                wk_choice = np.random.randint(0,wk,nsols)
                #print(wk_choice)

                it_choice = np.random.randint(0,it,nsols)
                #print(it_choice)

                cube_params = [ chains[w,i,:] for w,i in zip(wk_choice,it_choice) ]
                flat_params = [cube_to_physical(self.PSystem, cp) for cp in cube_params]


            elif mode.lower() == 'best':
                print('best')
                # best_solutions comes from better to worse
                best_solutions = extract_best_solutions(self.hdf5_file, 
                                                        write_file=False)
                
                cube_params = [list(bs[1]) for bs in best_solutions[:nsols] ]

                flat_params = [cube_to_physical(self.PSystem, cp) for cp in cube_params]

            else:                
                raise SystemExit('Impossible to understand -mode- argument. '+\
                    "Valid options are: 'random', 'best' ") # ?

            """
            # TODO: allow flat_params be specified by user
            if type(flat_params) in (list, np.array):

                # Check if flat_params are equal to 7*NPLA
                # Esto debe estar en unidades fisicas
                if len(flat_params) == 7*self.PSystem.NPLA:
                    flat_params = list(flat_params)
                else:
                    for fp in flat_params:
                        assert fp == 7 * self.PSystem.NPLA, "flat_params items "+\
                    "must have len = 7 * number of planets."

                    #assert flat_params == 7 * self.PSystem.NPLA, "flat_params items "+\
                    #"must have len = 7 * number of planets."
            """
        else:
            # No data to plot. Just the observed TTVs.
            print('---> No data to plot')


        # =====================================================================
        """
        if len(flat_params)>1:
            ndata = 1 #1
            flat_params = flat_params[::ndata]

            # truco
            logl =  flat_params[:,0] + abs(min(flat_params[:,0])) + 1.
        
            bests = flat_params[:,1:]
            flat_params = np.array([cube_to_physical(self.PSystem, x) for x in bests] )
            flat_params = np.array([_remove_constants(self.PSystem, x) for x in flat_params])
            #
            # Reversa
            logl = logl[::-1]
            flat_params = flat_params[::-1]
            #

            norm = matplotlib.colors.LogNorm(
            vmin=np.min(logl),
            vmax=np.max(logl))
            
            c_m = matplotlib.cm.gnuplot_r

            s_m = matplotlib.cm.ScalarMappable(cmap=c_m, norm=norm)
            s_m.set_array([])
        """
        # BEGINS FIGURE

        # FIXME: There's a bug when the number of simulated transits doesn't 
        # coincide with the number of observations
        if nplots > 1:
            fig, axes = plt.subplots(figsize=self.size, nrows=nplots, ncols=1, sharex=True)
        else: 
            fig, axes = plt.subplots(figsize=self.size)

        # X limits
        l_xlim, r_xlim = self.PSystem.T0JD-1, self.PSystem.Ftime+1

        index = 0

        # Iterates over TTVs of individual Planets
        for planet_id in self.PSystem.TTVs.keys(): 

            if hasattr(self.PSystem.planets[planet_id], "ttvs_data"):
                # Read observed TTVs of current planet
                #ttvs_dict = {k:self.PSystem.TTVs[planet_id][k] for k 
                #                        in sorted(self.PSystem.TTVs[planet_id].keys())}
                ttvs_dict = self.PSystem.TTVs[planet_id]

                errors = np.array([[v[1]*mins, v[2]*mins] for k,v in ttvs_dict.items()]).T
                
                # Make a model O-C given by the transits
                ttvs_dict = {k:v[0] for k,v in ttvs_dict.items()}                                
                x_obs, y_obs, model_obs = self._calculate_model(ttvs_dict)

                # More than one subplot?
                if nplots > 1:
                    ax = axes[index]
                else:
                    ax = axes

                ax.set_ylabel("O-C [min]")


                # Plot observed TTVs
                if show_obs:
                    ax.errorbar(y_obs, 
                                (y_obs-model_obs.predict(x_obs))*mins, 
                                yerr=errors, 
                                color=self.colors[index], 
                                ecolor='gray',
                                fmt='o', 
                                markersize=4,
                                mec='k',
                                mew=0.5,
                                alpha=1,
                                label=f'{planet_id}',
                                barsabove=False)

                    #sns.scatterplot(x=y_obs, 
                    #                y=(y_obs-model_obs.predict(x_obs))*mins, 
                    #                marker="o", ax=ax, color=self.colors[index], 
                    #                s=4,alpha=1. ,zorder=100000)

                    ax.set_xlim(l_xlim, r_xlim)


                # Make space for residuals
                if residuals:
                    divider = make_axes_locatable(ax)
                    sub_ax = divider.append_axes("bottom", size="30%", pad=0.1)
                    sub_ax.axhline(0, alpha=0.3, color='k')
                    sub_ax.set_xlim(l_xlim, r_xlim)
                    # Turn off xticklabels in main figure
                    ax.set_xticklabels([])

                # Iterate over solutions in flat_params
                if flat_params != None:

                    for isol, solution in enumerate(flat_params):                                       

                        # Perform the simulation for the current solution
                        SP = run_TTVFast(solution,  
                                    mstar=self.PSystem.mstar,
                                    init_time=self.PSystem.T0JD, 
                                    final_time=self.PSystem.Ftime, 
                                    dt=self.PSystem.dt)                    

                        EPOCHS = calculate_ephemeris(self.PSystem, SP)
                        
                        # Make coincide the number of observed and simulated transits
                        epochs = {epoch[0]:EPOCHS[planet_id][epoch[0]] for epoch in x_obs }

                        # model
                        x_cal, y_cal, model_calc = self._calculate_model(epochs)
                        
                        # Plot O-C
                        ax.plot(y_cal, 
                                (y_cal-model_calc.predict(x_cal))*mins , 
                                color= self.colors[index] ,
                                lw=1.2, #0.5,
                                alpha=1,
                                )
                        ax.set_xlim(l_xlim, r_xlim)

                        
                        # Plot residuals
                        if residuals:
                            #
                            residual_obs = {x:(y-model_obs.predict( np.array([x]).reshape(1,-1) ))*mins  for x,y in zip(list(x_obs.flatten()), list(y_obs))}
                            residual_cal = {x:(y-model_calc.predict( np.array([x]).reshape(1,-1) ))*mins for x,y in zip(list(x_cal.flatten()), list(y_cal))}

                            residuals = [residual_obs[r]-residual_cal[r] for r in sorted(ttvs_dict.keys())]
                            
                            sub_ax.scatter(y_obs, residuals, 
                                            s=2, color=self.colors[index],
                                            alpha=1.)
                            sub_ax.set_xlim(l_xlim, r_xlim)


                ax.legend(loc='upper right')
                index += 1

            else:
                print(f"No ttvs have been provided for planet {planet_id}")

        plt.xlabel(f"Time [days]")
        plt.tight_layout()
        plt.subplots_adjust(top=0.95)
       
        return 


    def plot_hist(self,  chains=None, titles=False):
        # TODO: Include a burnin line for chains!!!!
        # TODO TODO TODO TODO !!!
        """Make histograms of the every planetary parameter
        
        Arguments:
            chains {array} -- An array with shape (nwalkers, niters, nparams)
        """
        
        assert(0.0 <= self.burnin <= 1.0), f"burnin must be between 0 and 1!"
        if chains is not None:
            assert(len(chains.shape) == 3), "Shape for chains should be: (walkers,steps,dim)"+\
                f" instead of {chains.shape}"
            nwalkers,  it, _ = chains.shape
            burnin_ = int(self.burnin*(it) )
            chains  = chains[:,burnin_:,:]   

        elif self.hdf5_file:
            assert(isinstance(self.hdf5_file, str) ), "self.hdf5_file must be a string"
            # Extract chains from hdf5 file
            f = h5py.File(self.hdf5_file, 'r')
            index = f['INDEX'].value[0]
            chains = f['CHAINS'].value[self.temperature,:,:,:]
            nwalkers = f['NWALKERS'][0]
            f.close()

            burnin = int(self.burnin*index)
            #last_it = int(converge_time / intra_steps)
            chains = chains[:,burnin:index+1,:]
        
        else:
            raise RuntimeError("No chains or hdf5 file specified")

        ##
        # Convert from  normalized to physical
        chains = np.array([[cube_to_physical(self.PSystem, x) for x in chains[nw,:,:]] for nw in range(nwalkers) ])
        ##

        # Figure
        sns.set(context=self.sns_context,style=self.sns_style)
        _, axes = plt.subplots(nrows=self.PSystem.NPLA, ncols=7, 
                                figsize=(16,3*self.PSystem.NPLA))

        dim = 0
        for n in range(self.PSystem.NPLA):
            for p in range(7):
                param_idx = (n*7) + p

                # Write labels
                if n == self.PSystem.NPLA-1:
                    axes[n, p].set_xlabel(labels[p]+" "+units_latex[p], labelpad=10)
                # Write planet names
                if p == 0:
                    planet_ID = [k for k,v in self.PSystem.planets_IDs.items() if v==n]
                    axes[n, p].set_ylabel(planet_ID[0], )

                # Plot histograms
                if param_idx not in list(self.PSystem.constant_params.keys()):
                    parameter = chains[:,:,dim].flatten()
                    sns.distplot(parameter, kde=False, hist=True, 
                                color=colors[p], ax=axes[n, p], bins=20)
                    low, med, up = np.percentile(parameter, [16,50,84])
                    
                    if p == 1: 
                        # For period increase decimals
                        tit = r"$\mathrm{ %s ^{+%s}_{-%s} }$" % (round(med,4),
                                                        round(up-med,4),
                                                        round(med-low,4))
                    elif p == 2:
                        # For eccentricity increase decimals
                        tit = r"$\mathrm{ %s ^{+%s}_{-%s} }$" % (round(med,3),
                                                        round(up-med,3),
                                                        round(med-low,3))
                    else:
                        tit = r"$\mathrm{ %s ^{+%s}_{-%s} }$" % (round(med,2),
                                                        round(up-med,2),
                                                        round(med-low,2))
                    if titles:
                        axes[n, p].set_title(tit)
                    
                    axes[n, p].set_yticks([])
                    dim += 1
                
                # Write in title the constant values
                else:
                    if titles:
                        axes[n, p].set_title("{}".format(self.PSystem.constant_params[param_idx]))
                    axes[n, p].set_yticks([])
                    dim += 1
                
        plt.tight_layout()
        plt.subplots_adjust(wspace=0.15, hspace=0.3)
        
        return


    def plot_chains(self, chains=None, plot_means=False, thin=1):
        """[summary]
        
        Keyword Arguments:
            chains {[type]} -- [description] (default: {None})
            hdf5_file {[type]} -- [description] (default: {None})
            plot_means {bool} -- If True, then plot the mean of the chains at each
                        iteration for all the dimensions (default: {True})
        """
        xlabel = 'Iteration / intra_steps '
        if chains is not None:
            index = chains[::int(thin),:,:].shape[1] # The length of the chain

        if self.hdf5_file:
            # Extract chains from hdf5 file
            f = h5py.File(self.hdf5_file, 'r')
            index = f['INDEX'].value[0]
            #nwalkers = f['NWALKERS'][0]
            # shape for chains is: (temps,walkers,steps,dim)
            chains = f['CHAINS'].value[self.temperature,::int(thin),:index+1,:]
            total_walkers = chains.shape[0]

            intra_steps = f['INTRA_STEPS'].value[0]
            f.close()


            xlabel = f'Iteration / {intra_steps} '
            ##
            chains = np.array([[cube_to_physical(self.PSystem, x) for x in chains[nw,:,:]] for nw in range(total_walkers) ])
            ##

        sns.set(context=self.sns_context,style=self.sns_style)
        nrows = self.PSystem.NPLA*7 - len(self.PSystem.constant_params)
        _, axes = plt.subplots(nrows=nrows, ncols=1, # len(self.bounds)
                                figsize=(20, 8*self.PSystem.NPLA), sharex=True)

        dim = 0
        for pla in range(self.PSystem.NPLA):
            for param in range(7):
                param_idx = (pla*7) + param

                if param_idx not in list(self.PSystem.constant_params.keys()):

                    axes[dim].plot( chains[:,:,param_idx].T,   # chains[::int(thin),:,param_idx].T 
                            color=colors[param], alpha=0.1)

                    axes[dim].set_ylabel(labels[param]+str(pla+1)+ "\n" + 
                                                units_latex[param], labelpad=10)
                    #dim += 1
                    # Plot means
                    if plot_means:
                        means  =[np.median(chains[:,it,param_idx].T) for it in range(index)]
                        axes[dim].plot(means, c='k' ) 
                    dim += 1
                #dim += 1
        axes[dim-1].set_xlabel(xlabel)

        return


    def plot_corner(self, chains=None, color='#0880DE',titles=False):
        """[summary]
        
        Arguments:
            h5name {[type]} -- [description]
        
        Keyword Arguments:
            burnin {float} -- A number between 0 and 1 that represents the
                initial fraction of burnin of the total of iterations, e.g., 
                burnin = 0.2, removes the initial 20% of the chains.
        """

        assert(0.0 <= self.burnin <= 1.0), f"burnin must be between 0 and 1!"

        ##ndim = len(self.bounds) #npla*7
        #colnames = self.params_names.split()

        if self.hdf5_file:
            f = h5py.File(self.hdf5_file, 'r')
            index = f['INDEX'].value[0]
            intra_steps = f['INTRA_STEPS'].value[0]
            converge_time = f['ITER_LAST'].value[0]
            chains = f['CHAINS'].value[:,:,:index+1,:]
            nwalkers= f['NWALKERS'].value[0]
            #names_params = f["COL_NAMES"]
            f.close()

            burnin = int(self.burnin*index)
            last_it = int(converge_time / intra_steps)
            
            chains = chains[0,:,burnin:last_it,:]
            ##
            chains = np.array([[cube_to_physical(self.PSystem, x) for x in chains[nw,:,:]] for nw in range(nwalkers) ])
            ##
            

        # FIXME PARECE QUE AQUI FALTA CONSIDERAR index PARA HACER EL BURNIN SOBRE chains
        nwalkers = chains.shape[0]
        steps = chains.shape[1]
        # nuevo
        chains = np.array([[_remove_constants(self.PSystem, x) for x in chains[nw,:,:]] for nw in range(nwalkers) ])
        #
        chains_2D = chains.reshape(nwalkers*steps,self.PSystem.ndim) 

        """
        # Rename the columns of the data frame
        colnames = []
        for p in range(npla): 
            for _,v in labels.items(): 
                colnames.append(v+str(p+1))
        """
        
        """colnames = []
        for pla in range(self.NPLA):
            for param in range(7):
                param_idx = (pla*7) + param

                if param_idx not in list(self.constant_params.keys()):
                    colnames.append("\n" + labels[param]+str(pla+1) + "\n" )
        """
        colnames = self.PSystem.params_names.split() # nuevo
        df = pd.DataFrame(chains_2D, columns=colnames)

        """
        # Detect which columns have constant values and drop them
        cols_to_drop = []
        for (colname, coldata) in df.iteritems():
            if np.std(coldata) < 1e-8:
                cols_to_drop.append(colname)

        if len(cols_to_drop) > 0:
            df.drop(columns=cols_to_drop, inplace=True)
        """

        sns.set(context=self.sns_context,style=self.sns_style,
                font_scale=0.5,rc={"lines.linewidth": 0.5})

        corner.corner(df, quantiles=[0.16, 0.5, 0.84], 
                    show_titles=titles, 
                    title_kwargs={"fontsize": 12}, 
                    title_fmt='.3f',
                    label_kwargs={"fontsize": 12, "labelpad": 20},
                    plot_contours=True, 
                    plot_datapoints=False, 
                    plot_density=True, 
                    #data_kwargs={'markersize':3,'alpha':0.005, 'lw':10} , 
                    color = color,
                    smooth=True,
                    hist_kwargs={'color':'k', 'rwidth':0.8}
                    )

        #plt.gcf().set_size_inches(12,12)
        plt.subplots_adjust(
        top=0.969,
        bottom=0.061,
        left=0.056,
        right=0.954,
        wspace=0.015,
        hspace=0.015)
        
        
        return


    def plot_monitor(self):

        sns.set(context=self.sns_context,style=self.sns_style)

        f = h5py.File(self.hdf5_file, 'r')
        bestlogl = f['BESTLOGL'].value
        betas = f['BETAS'].value
        acc = f['ACC_FRAC0'].value
        meanlogl = f['MEANLOGL'].value
        tau = f['AUTOCORR'].value
        conv = f['INTRA_STEPS'].value[0]
        index = f['INDEX'].value[0]
        f.close()

        _, axes = plt.subplots(nrows=2, ncols=2, figsize=(20,10), sharex=True)

        # Temperatures
        axes[0,0].plot(1./betas)
        axes[0,0].plot(1./betas[:,0], color='k', lw=3)
        axes[0,0].set_ylabel("Temperature")
        axes[0,0].set_yscale("symlog")

        # swap
        axes[0,1].plot(acc[:index+1])
        axes[0,1].plot(acc[:index+1,0], color='k', lw=3)
        axes[0,1].set_ylabel("Accepted temperature swap fractions")

        # Loglikelihood
        axes[1,0].plot(meanlogl[:index+1], label='Mean Loglikelihood')
        axes[1,0].plot(bestlogl[:index+1], label='Maximum Loglikelihood')
        axes[1,0].set_ylabel("Log likelihood")
        axes[1,0].set_yscale("symlog")
        axes[1,0].set_xlabel(f"Steps/{conv}")
        axes[1,0].legend(loc='lower right')

        # Tau
        axes[1,1].plot(tau[:index+1])
        axes[1,1].set_ylabel("Mean autocorrelation time")
        #x = np.linspace(0, index+1, 20)
        #axes[1,1].plot(x, x*50/conv, '--k')
        axes[1,1].set_xlabel(f"Steps/{conv}")

        plt.subplots_adjust(wspace=0.15, hspace=0.05)

        return



    def plot_convergence(self,  chains=None, names=None, nchunks_gr=10, thinning=1):

        assert(0.0 <= self.burnin <= 1.0), f"burnin must be between 0 and 1!"

        if chains is not None:
            assert(len(chains.shape)==3), "Shape for chains should be:"+\
                f" (walkers,steps,dim) instead of {chains.shape}"
            if names is not None:
                pass
            else:
                # Create a generic list of names
                names = [f"dim{d}" for d in list(range(chains.shape[-1]))] 

        elif self.hdf5_file:
            f = h5py.File(self.hdf5_file, 'r')
            chains = f['CHAINS'].value[0,:,:,:]
            names = list(f['COL_NAMES'].value[:].split())
            f.close()

        else:
            raise RuntimeError("No chains or hdf5 file specified")
        
        print("-- ", chains.shape)

        
        _, axes = plt.subplots(nrows=2, ncols=1, figsize=(20,10))
        symbols = {1:"o",2:"^",3:"x",4:"s",5:"P"}
        

        # Gelman Rubin test
        GR = gelman_rubin(chains=chains, nchunks_gr=nchunks_gr, thinning=thinning, names=names)

        # Select the steps to perform GR statistic 
        steps = [ int(it) for it in np.linspace(0,chains.shape[1],nchunks_gr+1)[:-1] ]

        i=0
        m=1
        for k, v in GR.items():
            if i==10:
                m+=1
                i=0
            axes[0].plot(steps, v, marker=symbols[m], alpha=0.5, label=k)
            i+=1

        axes[0].set_title(f"{self.PSystem.system_name}", fontsize=15)
        axes[0].axhline( 1.01, color='gray', alpha=0.8, ls="--")
        axes[0].axhline( 1, color='k', alpha = 0.5, ls='--')
        axes[0].set_xlabel("Start iteration", fontsize=15)
        axes[0].set_ylabel(r'$\hat{R}$', fontsize=15)
        axes[0].legend(loc='upper left', fontsize=8) #bbox_to_anchor=(1.04,1), loc="bottom left")
        axes[0].grid(alpha=0.1)    
            
        # Geweke test
        Z = geweke(chains=chains, names=names, burnin=self.burnin)
        
        i=0
        m=1
        for k, v in Z.items():
            if i==10:
                m+=1
                i=0
            axes[1].plot(v, marker=symbols[m], alpha=0.5, label=k) 
            i+=1

        axes[1].axhline(-1, color='gray', alpha=0.5, ls="--")
        axes[1].axhline( 1, color='gray', alpha=0.5, ls="--")
        axes[1].axhline( 0, color='k', alpha = 0.8, ls='--')
        axes[1].axhline(-2, color='gray', alpha=0.7, ls="--")
        axes[1].axhline( 2, color='gray', alpha=0.7, ls="--")
        axes[1].set_xlabel("Second 50% of the samples (20 chunks)", fontsize=15)
        axes[1].set_ylabel("Z-score", fontsize=15)
        axes[1].grid(alpha=0.1)    
        
        
        plt.subplots_adjust(hspace=0.2)
        
        return {"GR": GR, "Z": Z}






    @staticmethod
    def _calculate_model(ttvs_dict):
        X, Y = [], []
        for k, v in ttvs_dict.items():
            X.append(k)
            Y.append(v)
        
        X = np.array(X).reshape((-1, 1))
        Y = np.array(Y)
        
        model = LinearRegression().fit(X, Y)
        #b, m = model.intercept_, model.coef_

        return X, Y, model


    def _check_dimensions(self, fp):
        # Verify that solution contains ALL the required parameters
        # to make the simulation. If don't, complete the fields
        # with the constant parameters. fp must be list.
        if len(fp) == self.PSystem.NPLA*7 and len(self.PSystem.constant_params)!=0:
            # there are the correct number of dimensions, but should be more?
            print("1")
            raise ValueError('Invalid values in flat_params. Pass just planetary parameters')
            
        elif len(fp) == self.PSystem.NPLA*7 and len(self.PSystem.constant_params)==0:
            # dimensions are correct
            print("2")
            return fp
        else:
            print("3")
            # insert constant params
            for k, v in self.PSystem.constant_params.items(): 
                fp.insert(k, v)
            return fp

