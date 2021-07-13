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
from dataclasses import dataclass, field
from .constants import colors, units_latex, labels


__all__ = ["Plots"]

__doc__ = "Built-in figures to visualize the main results from nauyaca"


@dataclass
class Plots:
    """A collection of customizable predefined figures.

    Use this module to visualize the main results from nauyca. The functions
    use the information from the hdf5 file (mcmc results) or from planet
    parameters specified by user.

    Parameters
    ----------
    PSystem : 
        The Planetary System object
    hdf5_file : str, optional
        The name of the hdf5 file from which the results will be extracted, by 
        default None. If given, specify the burnin.
    temperature : int, optional
        The target temperature (belonging to the ladder) that will be plotted, 
        by default 0. Note that temperatures different from 0 are support
        temperatures, so neither conclusion must come from them.
    burnin : float, optional
        A fraction between 0 and 1 to discard as burn-in at the beggining of
        the chains, by default 0.0 (no burning). This burnin will be applied 
        over the chains extracted from the hdf5 file.        
    sns_context : str, optional
        The 'context' kwarg in seaborn.set(), by default 'notebook'. It 
        modifies the size of labels, lines and other elements of the plot. 
        Options are: 'notebook', 'paper', 'talk', 'poster'. See seaborn doc.
    sns_style : str, optional
        The 'style' kwarg in seaborn.set(), by default 'ticks'. It changes
        the color axes, grids, etc.
    sns_font : float, optional
        The 'font_scale' kwarg in seaborn.set(), by default 1. It scales the
        font size of the axes.
    colors : dict, optional
        A dictionary to color lines, distributions, markers, etc. The values
        must be integers starting from 0 up to 6 (at least), and values must be
        the color names or hexadecimal code (eg., 'red', 'k', '#E5AC2F'). By
        default, it is taken from nauyaca.constants.colors
    """


    PSystem : None
    hdf5_file : str = None
    temperature : int = 0
    burnin : float = 0.0  
    sns_context : str = 'notebook'
    sns_style : str = 'ticks'
    sns_font : float = 1
    colors : dict = field(default_factory=lambda: colors)


    def TTVs(self, flat_params=None, nsols=1, mode='random', show_obs=True, 
                residuals=True, size=(10,10), line_kwargs={'alpha':0.5},
                xlabel="Time [days]"):
        """Plot the TTVs from the hdf5_file class instance, or from the
        specified flat_params.

        Parameters
        ----------
        flat_params : array or list
            A flat array containing the seven planet parameters of the first planet
            concatenated with the seven parameters of the next planet and so on. 
            The order per planet must be: 
            mass [Mearth], period [days], eccentricity, inclination [deg], argument
            of periastron [deg], mean anomaly [deg] and ascending node [deg].
            It can be a list of lists to plot many solutions. If individual
            solutions do not include the constant parameters (for example, if
            flat_params is loaded from a *_phys.opt file), they are added from
            PSystem.constant_params dictionary.
        nsols : int, optional
            The number of solutions to take from the posteriors, by default 1.
        mode : str, optional
            If the class instance hdf5_file is provided, this argument indicates
            the mode the solutions will be taken from the posteriors (after the
            burnin phase), by default 'random'. Two modes are available:
            * 'random': it takes random samples from the posteriors until 
                    complete nsols.
            * 'best': it takes the best solutions from the posteriors until
                    coplete nsols.
        show_obs : bool, optional
            A flag to indicate whether data points are plotted or not, by 
            default True.
        residuals : bool, optional
            A flag to indicate whether residuals are plotted in the figure or 
            not, by default True.
        size : tuple, optional
            The figure size, by default (10,10)
        line_kwargs : dict, optional
            A dictionary to pass **kwargs for the lines, by default {'alpha':0.5}
        xlabel : str, optional
            horizontal label of the figure, by default "Time [days]"
        """        

        mins = 1440.
        sns.set(context=self.sns_context, style=self.sns_style, font_scale=self.sns_font)
        nplots = len(self.PSystem.TTVs)

        # =====================================================================
        # PREPARING DATA
        # flat_params have to:
        #   * be a list of lists!
        #   * be in physical values with all the dimensions

        try:
            flat_params = list(flat_params)
        except:
            pass


        if flat_params != None:
            # Here, flat_params must be in physical values!

            # Is a unique list or array
            if isinstance(flat_params[0], float):
                flat_params = list(flat_params)  # convert to list
                flat_params = self._check_dimensions(flat_params)
                flat_params = [flat_params]   # convert to list of list

            # Items are iterables
            elif isinstance(flat_params[0], Iterable):
                flat_params = [list(fp) for fp in flat_params ]
                flat_params = [self._check_dimensions(fp) for fp in flat_params]
                
            else: 
                raise ValueError("Invalid items in flat_params. Items must be"+
                                    " planet parameters or list of solutions")


        elif self.hdf5_file != None:
            # Try to build flat_params from hdf5 and the attributes of this function
            # Here, flat_params can be in the normalized or physical form.

            if mode.lower() == 'random':
                #FIXME: There is a bug when random mode is active:
                # Sometimes a random solution without enough data is selected,
                # Producing KeyError
                print("--> plotting random solutions")

                r = get_mcmc_results(self.hdf5_file, keywords=['INDEX','CHAINS'])
                index = int(r['INDEX'][0])
                burnin_ = int(self.burnin*(index+1))
                chains  = r['CHAINS'][0,:,burnin_:index+1,:]
                wk, it, _ = chains.shape 
                del r 

                # Random choice
                wk_choice = np.random.randint(0,wk,nsols)
                it_choice = np.random.randint(0,it,nsols)

                rdm_params = [ list(chains[w,i,:]) for w,i in zip(wk_choice,it_choice) ]
                #flat_params = [cube_to_physical(self.PSystem, cp) for cp in rdm_params]

            elif mode.lower() == 'best':
                print('--> plotting best solutions')
                # best_solutions comes from better to worse
                best_solutions = extract_best_solutions(self.hdf5_file, 
                                                        write_file=False)
                
                rdm_params = [list(bs[1]) for bs in best_solutions[:nsols] ]
                #flat_params = [cube_to_physical(self.PSystem, cp) for cp in rdm_params]

            else:                
                raise SystemExit('Impossible to understand -mode- argument. '+\
                    "Valid options are: 'random', 'best' ") 


            # Decide if it have to be converted to physical values
            # Convert from  normalized to physical
            if (np.array(rdm_params) >= 0.).all() and (np.array(rdm_params) <= 1.).all():
                # Convert to physical values. It includes the constant parameters
                flat_params = [cube_to_physical(self.PSystem, rp) for rp in rdm_params]

            else:
                # Insert the constant parameters
                flat_params = []
                for rp in rdm_params:
                    for k, v in self.PSystem.constant_params.items():
                        rp.insert(k,v) 
                    flat_params.append(rp)
                  
        else:
            # No data to plot. Just the observed TTVs.
            print('---> No solutions to plot')
        

        # =====================================================================
        # BEGINS FIGURE

        # FIXME: There's a bug when the number of simulated transits doesn't 
        # coincide with the number of observations
        if nplots > 1:
            fig, axes = plt.subplots(figsize=size, nrows=nplots, ncols=1, sharex=True)
        else: 
            fig, axes = plt.subplots(figsize=size)

        # X limits defined from time span
        deltat = (self.PSystem.ftime - self.PSystem.t0) * 0.02
        l_xlim, r_xlim = self.PSystem.t0-deltat, self.PSystem.ftime+deltat

        index = 0

        # Iterates over TTVs of individual Planets
        for planet_id in self.PSystem.TTVs.keys(): 

            if hasattr(self.PSystem.planets[planet_id], "ttvs_data"):

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

                    ax.set_xlim(l_xlim, r_xlim)


                # Make space for residuals
                if residuals:
                    divider = make_axes_locatable(ax)
                    sub_ax = divider.append_axes("bottom", size="30%", pad=0.1)
                    sub_ax.axhline(0, alpha=0.3, color='k')
                    sub_ax.set_xlim(l_xlim, r_xlim)
                    sub_ax.set_ylabel('Residuals\n[min]')
                    # Turn off xticklabels in main figure
                    ax.set_xticklabels([])

                # Iterate over solutions in flat_params
                if flat_params != None:

                    for solution in flat_params:

                        # Perform the simulation for the current solution
                        SP = run_TTVFast(solution,  
                                    mstar=self.PSystem.mstar,
                                    init_time=self.PSystem.t0, 
                                    final_time=self.PSystem.ftime, 
                                    dt=self.PSystem.dt)                    

                        EPOCHS = _ephemeris(self.PSystem, SP)
                        
                        # Make coincide the number of observed and simulated transits
                        #   Does is it necessary? It's better to plot all the
                        #   data points
                        #epochs = {epoch[0]:EPOCHS[planet_id][epoch[0]] for epoch in x_obs }

                        # model
                        x_cal, y_cal, model_cal = self._calculate_model(EPOCHS[planet_id]) #epochs
                        
                        # Plot O-C
                        ax.plot(y_cal, 
                                (y_cal-model_cal.predict(x_cal))*mins , 
                                color= self.colors[index] ,
                                **line_kwargs
                                )
                        ax.set_xlim(l_xlim, r_xlim)
                        
                        # Plot residuals
                        if residuals:
                            #
                            residual_obs = {x:(y-model_obs.predict( np.array([x]).reshape(1,-1) ))*mins  for x,y in zip(list(x_obs.flatten()), list(y_obs))}
                            residual_cal = {x:(y-model_cal.predict( np.array([x]).reshape(1,-1) ))*mins for x,y in zip(list(x_cal.flatten()), list(y_cal))}

                            residuals = [residual_obs[r]-residual_cal[r] for r in sorted(ttvs_dict.keys())]
                            
                            sub_ax.scatter(y_obs, residuals, 
                                            s=2, color=self.colors[index],
                                            alpha=1.)
                            sub_ax.set_xlim(l_xlim, r_xlim)


                ax.legend(loc='upper right')
                index += 1

            else:
                print(f"No ttvs have been provided for planet {planet_id}")

        plt.xlabel(f"{xlabel}")
        plt.tight_layout()
        plt.subplots_adjust(top=0.95)
       
        return fig


    def hist(self,  chains=None, titles=False, size=None, hist_kwargs={}):
        """Make histograms of the a posteriori distributions for each 
        planetary parameter

        Parameters
        ----------
        chains : array, optional
            An array with shape (nwalkers, steps, ndim), by default None.
            Elements of the array must be normalized between 0 and 1.
            If the class attribute hdf5_file is provided, then the chains are 
            taken from the keyword 'CHAINS' from the hdf5 file. Note that the
            class atribute burnin is applied over the chains independently of
            the origin (chains kwarg or hdf5_file).
        titles : bool, optional
            A flag to specify whether medians and 1-sigma errors are plotted at
            the top of histograms, by default False
        size : tuple, optional
            The figure size, by default None, in which case the size is 
            automatically calculated
        hist_kwargs: dict
            A dictionary of keyword arguments to sns.distplot()
        """                

        if size is None:
            size=(16,3*self.PSystem.npla)
        
        assert(0.0 <= self.burnin <= 1.0), f"burnin must be between 0 and 1!"

        if chains is not None:
            assert(len(chains.shape) == 3), "Shape for chains should be: (walkers,steps,ndim)"+\
                f" instead of {chains.shape}"
            nwalkers,  index, _ = chains.shape
            #burnin_ = int(self.burnin*(index) )
            #chains  = chains[:,burnin_:,:]   

        elif self.hdf5_file:
            assert(isinstance(self.hdf5_file, str) ), "self.hdf5_file must be a string"
            # Extract chains from hdf5 file
            f = h5py.File(self.hdf5_file, 'r')
            index = f['INDEX'].value[0]
            chains = f['CHAINS'].value[self.temperature,:,:,:]
            nwalkers = f['NWALKERS'][0]
            f.close()

            #burnin = int(self.burnin*index)
            #last_it = int(converge_time / intra_steps)
            #chains = chains[:,burnin:index+1,:]

            # Convert from  normalized to physical
            #chains = np.array([[cube_to_physical(self.PSystem, x) for x in chains[nw,:,:]] for nw in range(nwalkers) ])
        
        else:
            raise RuntimeError("No chains or hdf5 file specified")

        # burnin phase
        burnin_ = int(self.burnin*(index) )
        chains  = chains[:,burnin_:index+1,:]   
        
        # Convert from  normalized to physical
        if (chains >= 0.).all() and (chains <= 1.).all():
            # Convert to physical values. It includes the constant parameters
            chains = np.array([[cube_to_physical(self.PSystem, x) for x in chains[nw,:,:]] for nw in range(nwalkers) ])
            
        else:
            # chains are in physical values
            if chains.shape[-1]==7*self.PSystem.npla:
                # Dimensions are complet (including constants).
                # No insertion needed
                pass
            else:
                # Insert the constant parameters
                for cp, val in self.PSystem.constant_params.items():
                    chains = np.insert(chains, cp, val, axis=2)
        ## Convert from  normalized to physical
        #chains = np.array([[cube_to_physical(self.PSystem, x) for x in chains[nw,:,:]] for nw in range(nwalkers) ])

        # Figure
        sns.set(context=self.sns_context,style=self.sns_style,font_scale=self.sns_font)

        fig, axes = plt.subplots(nrows=self.PSystem.npla, ncols=7, figsize=size)



        dim = 0
        for n in range(self.PSystem.npla):
            for p in range(7):
                param_idx = (n*7) + p

                # Write x labels
                if n == self.PSystem.npla-1:
                    axes[n, p].set_xlabel(labels[p]+" "+units_latex[p], labelpad=10)

                # Plot histograms (no constant parameters)
                if param_idx not in list(self.PSystem.constant_params.keys()):
                    
                    parameter = chains[:,:,dim].flatten()
                    
                    # Manage kwargs for histograms
                    default_hist_kwargs = {'kde': False, 'hist': True, 'color': self.colors[p],
                                      'bins': 20}
                    default_hist_kwargs.update(hist_kwargs)
                    # Plot
                    sns.distplot(parameter, ax=axes[n, p], **default_hist_kwargs)

                    axes[n, p].set_yticks([])
                    axes[n, p].set_ylabel("")

                    low, med, up = np.percentile(parameter, [16,50,84])

                    if p == 0:
                        # Write planet names
                        planet_ID = [k for k,v in self.PSystem.planets_IDs.items() if v==n]
                        axes[n, p].set_ylabel(planet_ID[0])

                    if p == 1: 
                        # For period increase decimals
                        tit = r"$\mathrm{ %s ^{+%s}_{-%s} }$" % (round(med,5),
                                                        round(up-med,5),
                                                        round(med-low,5))
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
                    

                    dim += 1
                
                # Constant parameters
                else:
                    if titles:
                        axes[n, p].set_title("{}".format(self.PSystem.constant_params[param_idx]))
                    axes[n, p].set_yticks([])
                    # Change the xticks:
                    axes[n, p].set_xlim(-1,1)
                    axes[n, p].set_xticks([0])
                    axes[n, p].set_xticklabels([f'{self.PSystem.constant_params[param_idx]}' ])
                    dim += 1
                
        plt.tight_layout()
        plt.subplots_adjust(wspace=0.15, hspace=0.35)
        
        return fig


    def trace_plot(self, chains=None, plot_means=False, thin=1, size=None):
        """Plot the mcmc chains along all the dimensions

        Parameters
        ----------
        chains : array, optional
            An array with shape (walkers,steps,ndim), by default None. If not
            given, then the chains are taken from the keyword 'CHAINS' from the
            hdf5 file.
        plot_means : bool, optional
            A flag to plot the mean of the chains at each iteration for all the
            dimensions, by default False
        thin : int, optional
            A thinning factor of the chains, by default 1. It is useful when
            plotting many chains
        size : tuple, optional
            The figure size, by default None, in which case the sieze is 
            automatically calculated
        """                

        if size is None:
            size=(20, 8*self.PSystem.npla)


        xlabel = 'Iteration / intra_steps '
        if chains is not None:
            #index = chains[::int(thin),:,:].shape[1] # The length of the chains
            assert(len(chains.shape) == 3), "Shape for chains should be: (walkers,steps,ndim)"+\
                f" instead of {chains.shape}"
            nwalkers,  index, _ = chains.shape

        elif self.hdf5_file:
            # Extract chains from hdf5 file
            f = h5py.File(self.hdf5_file, 'r')
            index = f['INDEX'].value[0]
            # shape for chains is: (temps,walkers,steps,dim)
            chains = f['CHAINS'].value[self.temperature,::int(thin),:index+1,:]
            #total_walkers = chains.shape[0]
            nwalkers = f['NWALKERS'][0]
            intra_steps = f['INTRA_STEPS'].value[0]
            f.close()

            xlabel = f'Iteration / {intra_steps} '
            ##
            #chains = np.array([[cube_to_physical(self.PSystem, x) for x in chains[nw,:,:]] for nw in range(total_walkers) ])
            ##
        else:
            raise RuntimeError("No chains or hdf5 file specified")

        # No burnin is required

        # Convert from  normalized to physical
        if (chains >= 0.).all() and (chains <= 1.).all():
            # Convert to physical values. It includes the constant parameters
            chains = np.array([[cube_to_physical(self.PSystem, x) for x in chains[nw,:,:]] for nw in range(nwalkers) ])

        else:
            # chains are in physical values
            if chains.shape[-1]==7*self.PSystem.npla:
                # Dimensions are complet (including constants).
                # No insertion needed
                pass
            else:
                # Insert the constant parameters
                for cp, val in self.PSystem.constant_params.items():
                    chains = np.insert(chains, cp, val, axis=2)


        sns.set(context=self.sns_context,style=self.sns_style,font_scale=self.sns_font)

        nrows = self.PSystem.npla*7 - len(self.PSystem.constant_params)
        
        fig, axes = plt.subplots(nrows=nrows, ncols=1, figsize=size, sharex=True)

        dim = 0
        for pla in range(self.PSystem.npla):
            for param in range(7):
                param_idx = (pla*7) + param

                if param_idx not in list(self.PSystem.constant_params.keys()):

                    axes[dim].plot( chains[:,:,param_idx].T, 
                            color=self.colors[param], alpha=0.1)

                    axes[dim].set_ylabel(labels[param]+str(pla+1)+ "\n" + 
                                                units_latex[param], 
                                                labelpad=10, rotation=45)

                    # Plot means
                    if plot_means:
                        means  =[np.median(chains[:,it,param_idx].T) for it in range(index)]
                        axes[dim].plot(means, c='k' ) 
                    dim += 1

        axes[dim-1].set_xlabel(xlabel)

        return fig


    def corner_plot(self, chains=None, color='#0880DE',titles=False, corner_kwargs={}):
        """A corner plot to visualize possible correlation between parameters

        This function uses the corner.py package. If you use it in your research,
        cite the proper attribution available at:
        https://corner.readthedocs.io/en/latest/index.html

        Parameters
        ----------
        chains : array, optional
            An array with shape (walkers,steps,ndim), by default None. If not
            given, then the chains are taken from the keyword 'CHAINS' from the
            hdf5 file.
        color : str, optional
            A color name or hexadecimal code (eg., 'red', 'k', '#E5AC2F'), by 
            default '#0880DE'. This is used as corner kwarg.
        titles : bool, optional
            A flag to put titles at top of individual distributions, by default
            False. This is used as corner kwarg.
        corner_kwargs : dict, optional
            Extra kwargs for corner.corner function, by default empty dictionary,
            in which case, a set of predefined kwargs are used.
        """        

        assert(0.0 <= self.burnin <= 1.0), f"burnin must be between 0 and 1!"

        if chains is not None:
            assert(len(chains.shape) == 3), "Shape for chains should be: (walkers,steps,ndim)"+\
                f" instead of {chains.shape}"
            nwalkers,  index, _ = chains.shape

        elif self.hdf5_file:
            f = h5py.File(self.hdf5_file, 'r')
            index = f['INDEX'].value[0]
            #intra_steps = f['INTRA_STEPS'].value[0]
            #converge_time = f['ITER_LAST'].value[0]
            chains = f['CHAINS'].value[self.temperature,:,:index+1,:]
            nwalkers= f['NWALKERS'].value[0]
            f.close()

            #burnin = int(self.burnin*index)
            #last_it = int(converge_time / intra_steps)
            #chains = chains[0,:,burnin:last_it,:]
            ##

            #chains = np.array([[cube_to_physical(self.PSystem, x) for x in chains[nw,:,:]] for nw in range(nwalkers) ])
            # remove constant params
            #chains = np.array([[_remove_constants(self.PSystem, x) for x in chains[nw,:,:]] for nw in range(nwalkers) ])

        else:
            raise RuntimeError("No chains or hdf5 file specified")
            
        # burnin phase
        burnin_ = int(self.burnin*(index) )
        chains  = chains[:,burnin_:index+1,:]  

        # Convert from  normalized to physical
        if (chains >= 0.).all() and (chains <= 1.).all():
            # Convert to physical values. It includes the constant parameters
            chains = np.array([[cube_to_physical(self.PSystem, x) for x in chains[nw,:,:]] for nw in range(nwalkers) ])
            # remove constant params
            chains = np.array([[_remove_constants(self.PSystem, x) for x in chains[nw,:,:]] for nw in range(nwalkers) ])
        else:
            pass
            # Insert the constant parameters
            #for cp, val in self.PSystem.constant_params.items():
            #    chains = np.insert(chains, cp, val, axis=2)


        nwalkers = chains.shape[0]
        steps = chains.shape[1]
        #
        chains_2D = chains.reshape(nwalkers*steps,self.PSystem.ndim) 

        # Make a data frame
        colnames = self.PSystem.params_names.split()
        df = pd.DataFrame(chains_2D, columns=colnames)


        sns.set(context=self.sns_context,style=self.sns_style,font_scale=self.sns_font)

        # Default kwargs for corner
        corner_corner_kwargs = {"quantiles": [0.16, 0.5, 0.84],
                                "show_titles": titles,
                                "title_kwargs": {"fontsize": 12},
                                "title_fmt": '.3f',
                                "label_kwargs": {"fontsize": 15, "labelpad": 25},
                                "plot_contours": True,
                                "plot_datapoints": True,
                                "plot_density": True,
                                "color": color,
                                "smooth": True,
                                "hist_kwargs": {'color':'k', 'rwidth':0.8}
                                }

        corner_corner_kwargs.update(corner_kwargs)

        fig = corner.corner(df,
                     **corner_corner_kwargs 
                    )

        plt.subplots_adjust(
        top=0.969,
        bottom=0.061,
        left=0.056,
        right=0.954,
        wspace=0.,#015,
        hspace=0.#015
        )
        
        return fig


    def monitor(self,size=(20,10)):
        """Plot a monitor of the mcmc performance

        It encompass the temperature adaptation, swap fractions, Loglikelihood
        improvement and mean correlation time.

        Parameters
        ----------
        size : tuple, optional
            The figure size, by default (20,10)
        """

        f = h5py.File(self.hdf5_file, 'r')
        bestlogpost = f['MAP'].value
        betas = f['BETAS'].value
        acc = f['ACC_FRAC0'].value
        meanlogpost = f['MEANLOGPOST'].value
        tau = f['AUTOCORR'].value
        conv = f['INTRA_STEPS'].value[0]
        index = f['INDEX'].value[0]
        f.close()

        sns.set(context=self.sns_context,style=self.sns_style,font_scale=self.sns_font)

        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=size, sharex=True)

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
        axes[1,0].plot(meanlogpost[:index+1], label='Mean Log-posterior')
        axes[1,0].plot(bestlogpost[:index+1], label='Maximum a posteriori')
        axes[1,0].set_ylabel("Probability")
        axes[1,0].set_yscale("symlog")
        axes[1,0].set_xlabel(f"Steps/{conv}")
        axes[1,0].legend(loc='lower right')

        # Tau
        axes[1,1].plot(tau[:index+1])
        axes[1,1].set_ylabel("Mean autocorrelation time")
        axes[1,1].set_xlabel(f"Steps/{conv}")

        plt.subplots_adjust(wspace=0.15, hspace=0.05)

        return fig


    def convergence(self,  chains=None, names=None, nchunks_gr=10, thinning=1,size=(20,10)):
        """Plot two convergence test, namely, Gelman-Rubin and Geweke

        For Gelman-Rubin (R), a valid value for convergence assessment is R < 1.01
        For Geweke, a valid value for convergence assessment is -1 < Z-score < 1

        Parameters
        ----------
        chains : array, optional
            An array with shape (walkers,steps,ndim), by default None. If not
            given, then the chains are taken from the keyword 'CHAINS' from the
            hdf5 file.
        names : list, optional
            A list with param names that should correspond to ndim, by default 
            None, in which case the names are taken from the current PSystem.
        nchunks_gr : int, optional
            Number of chunks to divide the chain steps in the Gelman-Rubin test,
            by default 10.
        thinning : int, optional
            A thining factor of the chains, by default 1. Useful with larger 
            chains
        size : tuple, optional
            The figure size, by default (20,10)

        """

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

        sns.set(context=self.sns_context,style=self.sns_style,font_scale=self.sns_font)        
        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=size)
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
        axes[0].legend(loc='upper left', fontsize=8)
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
        
        return fig


    @staticmethod
    def _calculate_model(ttvs_dict):
        """A help function to calculate coefficients of the linear regression
        in the TTVs"""
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
        """A helpful function to verify that solution contains ALL the required 
        parameters to make the simulation. If don't, complete the fields with the 
        constant parameters. fp must be list. This is used in TTVs plot function"""

        if len(fp) == self.PSystem.npla*7 and len(self.PSystem.constant_params)!=0:
            # there are the correct number of dimensions, but should be more?
            
            raise ValueError('Invalid values in flat_params. Pass only planetary parameters')
            
        elif len(fp) == self.PSystem.npla*7 and len(self.PSystem.constant_params)==0:
            # dimensions are correct
            
            return fp
        else:
            
            # insert constant params
            for k, v in self.PSystem.constant_params.items(): 
                fp.insert(k, v)
            return fp
            