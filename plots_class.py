import seaborn as sns
from sklearn.linear_model import LinearRegression
from .utils import *
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import corner 
import h5py
import pandas as pd
import numpy as np
from dataclasses import dataclass

from .constants import colors, units_latex, labels

__all__ = ["Plots_c"]

@dataclass
class Plots_c:


    PSystem : None
    hdf5_file : str = 'None'
    temperature : int = 0
    burning : float = 0.0  
    sns_context : str = 'notebook'
    sns_style : str = 'darkgrid'



    def plot_TTVs(self, flat_params=None, mode='None', nsols=1, show_obs=True, residuals=True):
        """Plot the observed TTVs signals and the transit solutions from flat_params
        
        Arguments:
            self {object} -- The Planetary System object.
            flat_params {Array} -- A flat array containing mass, period, eccentricity,
                inclination, argument, mean anomaly, ascending node for all planets.
                It could be included more than one solution.
        """

        # Be aware of the constant parameters
        mins = 1440.
        sns.set(context=self.sns_context,style=self.sns_style)
        nplots = len(self.PSystem.TTVs) #self.PSystem.NPLA #


        # FIXME: There's a bug when the number of simulated transits doesn't 
        # coincide with the number of observations
        if nplots > 1:
            fig, axes = plt.subplots(figsize=(10,10), nrows=nplots, ncols=1, sharex=True)
        else: 
            fig, axes = plt.subplots(figsize=(8,10))
        
        #sns.despine()
        
        # =============
        # Preparacion de datos
        if (flat_params != None) and (isinstance(flat_params[0], list)):
            print("E1")
            pass 
        elif (flat_params != None) and (isinstance(flat_params[0], list)==False):
            print("E2")
            flat_params = [flat_params]

        else:
            print("E3")
            if mode.lower() == 'random':
                #FIXME: There is a bug when random mode is active:
                # Sometimes a random solution without enough data is selected,
                # Producing KeyError
                print("random")

                r = get_mcmc_results(self.hdf5_file, keywords=['INDEX','CHAINS'])
                index = int(r['INDEX'][0])
                burning_ = int(self.burning*(index+1))
                chains  = r['CHAINS'][0,:,burning_:index+1,:]
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

        #===============
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


        index = 0
        #alpha_grad = 1./len(flat_params)
        #print(alpha_grad)
        for planet_id in self.PSystem.TTVs.keys(): #self.PSystem.planets_IDs.items():

            if hasattr(self.PSystem.planets[planet_id], "ttvs_data"):

                # Read observed TTVs of current planet
                ttvs_dict = {k:self.PSystem.TTVs[planet_id][k] for k 
                                        in sorted(self.PSystem.TTVs[planet_id].keys())}
                # Tal vez lo anterior se puede cambiar por:
                # self.TTVs[planet_id]
                errors = np.array([[v[1]*mins, v[2]*mins] for k,v 
                                        in ttvs_dict.items()]).T
                
                # Make a model O-C given by the transits
                ttvs_dict = {k:v[0] for k,v in ttvs_dict.items()}                
                x_obs, y_obs, model_obs = self._calculate_model(ttvs_dict)
                #residuals_1 = {x:y for x,y in zip(ttvs_dict.keys(), )} #(y_obs-model_obs.predict(x_obs))*mins
                
                # 
                if nplots > 1:
                    ax = axes[index]
                else:
                    ax = axes

                # aqui estaba
                if show_obs:
                    # Plot observed TTVs
                    
                    ax.errorbar(y_obs, 
                                (y_obs-model_obs.predict(x_obs))*mins, 
                                yerr=errors, 
                                color='k', 
                                ecolor='k',
                                fmt='o', 
                                markersize=4,
                                alpha=1,
                                label=f'{planet_id}',
                                barsabove=True)
                    sns.scatterplot(x=y_obs, 
                                    y=(y_obs-model_obs.predict(x_obs))*mins, 
                                    marker="o", ax=ax, color='white',
                                    s=3.9,alpha=1. ,zorder=100000)
                    
                                  
                # Make space for residuals
                #if len(flat_params)>0:
                if residuals:
                    divider = make_axes_locatable(ax)
                    sub_ax = divider.append_axes("bottom", size="30%", pad=0.1)

                for isol, solution in  enumerate(flat_params):

                    # Plot the solutions given in flat_params
                    
                    ##SP = run_TTVFast(solution, mstar=self.PSystem.mstar, 
                    ##        NPLA=self.PSystem.NPLA,Tin=0., Ftime= self.PSystem.time_span, 
                    ##        dt=self.PSystem.dt )
                    SP = run_TTVFast(solution,  
                                mstar=self.PSystem.mstar, ##NPLA=PSystem.NPLA, 
                                #init_time=0., final_time=PSystem.time_span, 
                                init_time=self.PSystem.T0JD, 
                                final_time=self.PSystem.Ftime, 
                                dt=self.PSystem.dt)                    

                    """
                    EPOCHS = calculate_epochs(SP, self)
                    """
                    EPOCHS = calculate_ephemeris(SP, self.PSystem)
                    
                    # Make coincide the number of observed and simulated transits
                    epochs = {epoch[0]:EPOCHS[planet_id][epoch[0]] for epoch in x_obs }

                    # Whitout model
                    x_cal, y_cal, model_calc = self._calculate_model(epochs)
                    
                    ax.plot(y_cal, 
                            (y_cal-model_calc.predict(x_cal))*mins , 
                            color= colors[index] , #s_m.to_rgba( logl[isol]), 
                            lw=1.2, #0.5,
                            alpha=1,
                            ) #'-+'

                    
                    ax.set_ylabel("O-C [min]")
                    ##ax.set_xlim(self.PSystem.T0JD, self.PSystem.T0JD+self.PSystem.time_span)
                    ax.set_xlim(self.PSystem.T0JD-1, self.PSystem.Ftime+1)  # Provisional   

                    # =========== Para animacion
                    ##if index ==0:
                    ##    ax.set_ylim(-8,9)
                    ##if index ==1:
                    ##    ax.set_ylim(-4,4)
                    # ============
                    # TODO: Remove thicks in all figures except the last!!
                    
                    # Plot residuals
                    if residuals:
                        residuals_2 = {x:y for x,y in zip(list(x_cal.flatten()), list(y_cal))} #(y_cal-model_obs.predict(x_cal))*mins
                        #
                        residual_obs = {x:(y-model_obs.predict( np.array([x]).reshape(1,-1) ))*mins  for x,y in zip(list(x_obs.flatten()), list(y_obs))}
                        residual_cal = {x:(y-model_calc.predict( np.array([x]).reshape(1,-1) ))*mins for x,y in zip(list(x_cal.flatten()), list(y_cal))}
                        #OC_obs = (y_obs-model_obs.predict(x_obs))*mins 
                        #OC_cal = (y_cal-model_calc.predict(x_cal))*mins
                        #
                        ###residuals = [ r1-r2 for r1, r2 in 
                        ###                zip(residuals_1, residuals_2)]
                        #residuals = [(ttvs_dict[r]-residuals_2[r])*mins for r in sorted(ttvs_dict.keys()) ]
                        residuals = [residual_obs[r]-residual_cal[r] for r in sorted(ttvs_dict.keys())]
                        
                        sub_ax.scatter(y_obs, residuals, 
                                        s=2, color=colors[index],
                                        alpha=1.) #s_m.to_rgba( logl[isol]))

                        
                        ##sub_ax.set_xlim(self.PSystem.T0JD,self.PSystem.T0JD+self.PSystem.time_span)
                        sub_ax.set_xlim(self.PSystem.T0JD-1, self.PSystem.Ftime+1)
                #sub_ax.grid(alpha=0.5)
                
                sub_ax.axhline(0, alpha=0.3, color='k')
                #sub_ax.set_ylim(-3,3)

                ax.legend(loc='upper right')
                #ax.grid(alpha=0.5)
                index += 1

            else:
                print("No ttvs have been provided for planet {}".format(planet_id))

        plt.xlabel("Time [days]")
        plt.tight_layout()
        plt.subplots_adjust(top=0.95)

        #
        """
        if len(flat_params)>1:
            fig.subplots_adjust(right=0.8)
            cbar_ax = fig.add_axes([0.8, 0.15, 0.05, 0.7])
            #fig.colorbar(im, cax=cbar_ax)
            plt.colorbar(s_m, cax=cbar_ax).set_label('Probability')
        """
        #
        
        return fig


    def plot_hist(self,  chains=None):
        # TODO: Include a burning line for chains!!!!
        """Make histograms of the every planetary parameter
        
        Arguments:
            chains {array} -- An array with shape (nwalkers, niters, nparams)
        """
        
        assert(0.0 <= self.burning <= 1.0), f"burning must be between 0 and 1!"
        if chains is not None:
            assert(len(chains.shape) == 3), "Shape for chains should be: (walkers,steps,dim)"+\
                f" instead of {chains.shape}"
            nwalkers,  it, _ = chains.shape
            burning_ = int(self.burning*(it) )
            chains  = chains[:,burning_:,:]   

        if self.hdf5_file:
            assert(isinstance(self.hdf5_file, str) ), "self.hdf5_file must be a string"
            # Extract chains from hdf5 file
            f = h5py.File(self.hdf5_file, 'r')
            index = f['INDEX'].value[0]
            chains = f['CHAINS'].value[self.temperature,:,:,:]
            nwalkers = f['NWALKERS'][0]
            f.close()

            burning = int(self.burning*index)
            #last_it = int(converge_time / conver_steps)
            chains = chains[:,burning:index+1,:]

        ##
        # Convert from  normalized to physical
        chains = np.array([[cube_to_physical(self.PSystem, x) for x in chains[nw,:,:]] for nw in range(nwalkers) ])
        ##

        # Figura
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
                    axes[n, p].set_title(tit)
                    axes[n, p].set_yticks([])
                    dim += 1
                
                # Write in title the constant values
                else:
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
        xlabel = 'Iteration / conver_steps '
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

            conver_steps = f['CONVER_STEPS'].value[0]
            f.close()


            xlabel = f'Iteration / {conver_steps} '
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


    def plot_corner(self, chains=None, color='#0880DE'):
        """[summary]
        
        Arguments:
            h5name {[type]} -- [description]
        
        Keyword Arguments:
            burning {float} -- A number between 0 and 1 that represents the
                initial fraction of burning of the total of iterations, e.g., 
                burning = 0.2, removes the initial 20% of the chains.
        """

        assert(0.0 <= self.burning <= 1.0), f"burning must be between 0 and 1!"

        ###fisicos = np.array([[cube_to_physical(syn, x) for x in RESULTS_mcmc[0,nw,:,:]] for nw in range(Nwalkers) ])

        ##ndim = len(self.bounds) #npla*7
        #colnames = self.params_names.split()

        if self.hdf5_file:
            f = h5py.File(self.hdf5_file, 'r')
            index = f['INDEX'].value[0]
            conver_steps = f['CONVER_STEPS'].value[0]
            converge_time = f['ITER_LAST'].value[0]
            chains = f['CHAINS'].value[:,:,:index+1,:]
            nwalkers= f['NWALKERS'].value[0]
            #names_params = f["COL_NAMES"]
            f.close()

            burning = int(self.burning*index)
            last_it = int(converge_time / conver_steps)
            #print(burning, last_it)
            chains = chains[0,:,burning:last_it,:]
            ##
            chains = np.array([[cube_to_physical(self.PSystem, x) for x in chains[nw,:,:]] for nw in range(nwalkers) ])
            ##
            

        # PARECE QUE AQUI FALTA CONSIDERAR index PARA HACER EL BURNING SOBRE chains
        nwalkers = chains.shape[0]
        steps = chains.shape[1]
        # nuevo
        chains = np.array([[_remove_constants(self.PSystem, x) for x in chains[nw,:,:]] for nw in range(nwalkers) ])
        #
        chains_2D = chains.reshape(nwalkers*steps,self.PSystem.ndim)  #self.ndim CAMBIAR A len(BOUNDS)?

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
                    show_titles=True, 
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
        
        #plt.savefig('{}_corner.pdf'.format(h5name.split('.')[0]) )
        #plt.close()
        #plt.show()
        
        return


    def plot_monitor(self):

        sns.set(context=self.sns_context,style=self.sns_style)

        f = h5py.File(self.hdf5_file, 'r')
        bestlogl = f['BESTLOGL'].value
        betas = f['BETAS'].value
        acc = f['ACC_FRAC0'].value
        meanlogl = f['MEANLOGL'].value
        tau = f['AUTOCORR'].value
        conv = f['CONVER_STEPS'].value[0]
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

        # Chi2
        axes[1,0].plot(meanlogl[:index+1])
        axes[1,0].plot(bestlogl[:index+1])
        axes[1,0].set_ylabel("Log likelihood")
        axes[1,0].set_yscale("symlog")
        axes[1,0].set_xlabel(f"Steps/{conv}")

        # Tau
        axes[1,1].plot(tau[:index+1])
        axes[1,1].set_ylabel("Mean autocorrelation time")
        #x = np.linspace(0, index+1, 20)
        #axes[1,1].plot(x, x*50/conv, '--k')
        axes[1,1].set_xlabel(f"Steps/{conv}")

        plt.subplots_adjust(wspace=0.15, hspace=0.05)

        return


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

