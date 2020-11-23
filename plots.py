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

__doc__ = """This is a modificable script. Plots class is ran outside the 
            classes that make the TTVs operations
            """
__all__ = ["plot_TTVs", "plot_hist", "plot_chains", "plot_corner", 
            "plot_monitor", "plot_correl", "plot_convergence"]

colors = {0:"red", 1:"olive", 2:"skyblue", 3:"gold", 4:"teal",
          5:"orange", 6:"purple"}

labels = {0:r"$\mathrm{Mass}$" ,
           1:r"$\mathrm{Period}$",
           2:r"$\mathrm{eccentricity}$",
           3:r"$\mathrm{Inclination}$",
           4:r"$\mathrm{Argument\ (\omega)}$",
           5:r"$\mathrm{Mean\ anomaly}$",
           6:r"$\mathrm{\Omega}$"}

units = {0:r"$\mathrm{[M_{\oplus}]}$",
         1:r"$\mathrm{[days]}$",
         2:r"$\mathrm{ }$",
         3:r"$\mathrm{[deg]}$",
         4:r"$\mathrm{[deg]}$",
         5:r"$\mathrm{[deg]}$",
         6:r"$\mathrm{[deg]}$"}


def plot_TTVs(self, flat_params, show_obs=True):
    """Plot the observed TTVs signals and the transit solutions from flat_params
    
    Arguments:
        self {object} -- The Planetary System object.
        flat_params {Array} -- A flat array containing mass, period, eccentricity,
            inclination, argument, mean anomaly, ascending node for all planets.
            It could be included more than one solution.
    """

    # Be aware of the constant parameters

    mins = 1440.
    #sns.set_style("darkgrid")
    sns.set(context='paper',style='darkgrid')
    #sns.axes_style("darkgrid")
    #sns.set_style("darkgrid", {"axes.facecolor": ".9"})
    
    nplots = self.NPLA #len(self.TTVs)

    # FIXME: There's a problem when nplot=1, because ax=axes[index] doesn't
    # work: TypeError: 'AxesSubplot' object is not subscriptable
    # FIXME: There's a bug when the number of simulated transits doesn't 
    # coincide with the number of observations
    fig, axes = plt.subplots(figsize=(8,10), nrows=nplots, ncols=1, sharex=True)
    sns.despine()
    
    if len(flat_params)>1:
        ndata = 1 #1
        flat_params = flat_params[::ndata]

        # truco
        logl =  flat_params[:,0] + abs(min(flat_params[:,0])) + 1.
    
        bests = flat_params[:,1:]
        flat_params = np.array([cube_to_physical(self, x) for x in bests] )
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


    #

    index = 0
    for pids, _ in self.planets_IDs.items():

        if hasattr(self.planets[pids], "ttvs_data"):

            # Read observed TTVs of current planet
            ttvs_dict = {k:self.TTVs[pids][k] for k 
                                    in sorted(self.TTVs[pids].keys())}
            # Tal vez lo anterior se puede cambiar por:
            # self.TTVs[pids]
            errors = np.array([[v[1]*mins, v[2]*mins] for k,v 
                                    in ttvs_dict.items()]).T
            
            # Make a model O-C given by the transits
            ttvs_dict = {k:v[0] for k,v in ttvs_dict.items()}                
            x_obs, y_obs, model_obs = _calculate_model(ttvs_dict)
            #residuals_1 = {x:y for x,y in zip(ttvs_dict.keys(), )} #(y_obs-model_obs.predict(x_obs))*mins
            
            ax = axes[index]
            # aqui estaba
            if show_obs:
                # Plot observed TTVs
                sns.scatterplot(y_obs, (y_obs-model_obs.predict(x_obs))*mins, 
                                marker="o", ax=axes[index], color='k',
                                label=f'{pids}',s=5,alpha=1. ,zorder=100000)

                ax.errorbar(y_obs, (y_obs-model_obs.predict(x_obs))*mins, 
                                yerr=errors, color='gray', fmt='', 
                                alpha=0.5) #-1            

            # Make space for residuals
            if len(flat_params)>0:
                divider = make_axes_locatable(axes[index])
                sub_ax = divider.append_axes("bottom", size="30%", pad=0.1)

            for isol, solution in  enumerate(flat_params):
                # Plot the solutions given in flat_params
                SP = run_TTVFast(solution, mstar=self.mstar, 
                    NPLA=self.NPLA,Tin=0., Ftime= self.time_span, dt=self.dt )
                EPOCHS = calculate_epochs(SP, self)
                
                # Make coincide the number of observed and simulated transits
                epochs = {epoch[0]:EPOCHS[pids][epoch[0]] for epoch in x_obs }

                # Whitout model
                x_cal, y_cal, model_calc = _calculate_model(epochs)
                #print('calc: ', x_cal, y_cal)
                ax.plot(y_cal, 
                            (y_cal-model_calc.predict(x_cal))*mins , 
                            color=s_m.to_rgba( logl[isol]), 
                            lw=0.5
                            ) #'-+'
                #
                #print(logl[isol])
                #
                
                ax.set_ylabel("O-C [min]")
                ax.set_xlim(self.T0JD, self.T0JD+self.time_span )
                
                # TODO: Remove thicks in all figures except the last!!
                
                # Plot residuals
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
                sub_ax.scatter(y_obs, residuals, s=2, color=s_m.to_rgba( logl[isol]))
                sub_ax.set_xlim(self.T0JD, self.T0JD+self.time_span)
                ##sub_ax.grid(color='black')



            index += 1

        else:
            print("No ttvs have been provided for planet {}".format(pids))

    plt.xlabel("Time [days]")

    #
    if len(flat_params)>1:
        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.8, 0.15, 0.05, 0.7])
        #fig.colorbar(im, cax=cbar_ax)
        plt.colorbar(s_m, cax=cbar_ax).set_label('Probability')
    #
    
    return


def plot_hist(self, chains=None, hdf5_file=None, temperature=0, burning=0.0):
    # TODO: Includ a burning line for chains!!!!
    """Make histograms of the every planetary parameter
    
    Arguments:
        chains {array} -- An array with shape (nwalkers, niters, nparams)
    """
    
    assert(0.0 <= burning <= 1.0), f"burning must be between 0 and 1!"
    if chains is not None:
        assert(len(chains.shape) == 3), "Shape for chains should be: (walkers,steps,dim)"+\
            f" instead of {chains.shape}"
        nwalkers,  it, _ = chains.shape
        burning_ = int(burning*(it) )
        chains  = chains[:,burning_:,:]   

    if hdf5_file:
        assert(isinstance(hdf5_file, str) ), "hdf5_file must be a string"
        # Extract chains from hdf5 file
        f = h5py.File(hdf5_file, 'r')
        index = f['INDEX'].value[0]
        chains = f['CHAINS'].value[temperature,:,:,:]
        nwalkers = f['NWALKERS'][0]
        f.close()

        burning = int(burning*index)
        #last_it = int(converge_time / conver_steps)
        chains = chains[:,burning:index+1,:]

    ##
    # Convert from  normalized to physical
    chains = np.array([[cube_to_physical(self, x) for x in chains[nw,:,:]] for nw in range(nwalkers) ])
    ##

    # Figura
    sns.set(context='paper',style='white')
    _, axes = plt.subplots(nrows=self.NPLA, ncols=7, 
                            figsize=(16,3*self.NPLA))

    dim = 0
    for n in range(self.NPLA):
        for p in range(7):
            param_idx = (n*7) + p

            # Write labels
            if n == self.NPLA-1:
                axes[n, p].set_xlabel(labels[p]+" "+units[p], labelpad=10)
            # Write planet names
            if p == 0:
                planet_ID = [k for k,v in self.planets_IDs.items() if v==n]
                axes[n, p].set_ylabel(planet_ID[0], )

            # Plot histograms
            if param_idx not in list(self.constant_params.keys()):
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
                axes[n, p].set_title("{}".format(self.constant_params[param_idx]))
                axes[n, p].set_yticks([])
                dim += 1
            
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.15, hspace=0.3)
    
    return


def plot_chains(self, chains=None, hdf5_file=None, temperature=0, plot_means=False, thin=1):
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

    if hdf5_file:
        # Extract chains from hdf5 file
        f = h5py.File(hdf5_file, 'r')
        index = f['INDEX'].value[0]
        #nwalkers = f['NWALKERS'][0]
        # shape for chains is: (temps,walkers,steps,dim)
        chains = f['CHAINS'].value[temperature,::int(thin),:index+1,:]
        total_walkers = chains.shape[0]

        conver_steps = f['CONVER_STEPS'].value[0]
        f.close()

        #chains = chains[temperature,:,:index+1,:]
        xlabel = f'Iteration / {conver_steps} '
        ##
        chains = np.array([[cube_to_physical(self, x) for x in chains[nw,:,:]] for nw in range(total_walkers) ])
        ##

    sns.set(context='paper',style='white')
    nrows = self.NPLA*7 - len(self.constant_params)
    _, axes = plt.subplots(nrows=nrows, ncols=1, # len(self.bounds)
                            figsize=(20, 8*self.NPLA), sharex=True)

    dim = 0
    for pla in range(self.NPLA):
        for param in range(7):
            param_idx = (pla*7) + param

            if param_idx not in list(self.constant_params.keys()):

                axes[dim].plot( chains[:,:,param_idx].T,   # chains[::int(thin),:,param_idx].T 
                        color=colors[param], alpha=0.1)

                axes[dim].set_ylabel(labels[param]+str(pla+1)+ "\n" + 
                                            units[param], labelpad=10)
                #dim += 1
                # Plot means
                if plot_means:
                    means  =[np.median(chains[:,it,param_idx].T) for it in range(index)]
                    axes[dim].plot(means, c='k' ) 
                dim += 1
            #dim += 1
    axes[dim-1].set_xlabel(xlabel)

    return


def plot_corner(self, chains=None, hdf5_file=None, burning=0.0):
    """[summary]
    
    Arguments:
        h5name {[type]} -- [description]
    
    Keyword Arguments:
        burning {float} -- A number between 0 and 1 that represents the
            initial fraction of burning of the total of iterations, e.g., 
            burning = 0.2, removes the initial 20% of the chains.
    """

    assert(0.0 <= burning <= 1.0), f"burning must be between 0 and 1!"

    ###fisicos = np.array([[cube_to_physical(syn, x) for x in RESULTS_mcmc[0,nw,:,:]] for nw in range(Nwalkers) ])

    ##ndim = len(self.bounds) #npla*7
    #colnames = self.params_names.split()

    if hdf5_file:
        f = h5py.File(hdf5_file, 'r')
        index = f['INDEX'].value[0]
        conver_steps = f['CONVER_STEPS'].value[0]
        converge_time = f['ITER_LAST'].value[0]
        chains = f['CHAINS'].value[:,:,:index+1,:]
        nwalkers= f['NWALKERS'].value[0]
        names_params = f["COL_NAMES"]
        f.close()

        burning = int(burning*index)
        last_it = int(converge_time / conver_steps)
        #print(burning, last_it)
        chains = chains[0,:,burning:last_it,:]
        ##
        chains = np.array([[cube_to_physical(self, x) for x in chains[nw,:,:]] for nw in range(nwalkers) ])
        ##
        

    # PARECE QUE AQUI FALTA CONSIDERAR index PARA HACER EL BURNING SOBRE chains
    nwalkers = chains.shape[0]
    steps = chains.shape[1]
    # nuevo
    chains = np.array([[_remove_constants(self, x) for x in chains[nw,:,:]] for nw in range(nwalkers) ])
    #
    chains_2D = chains.reshape(nwalkers*steps,self.ndim)  #self.ndim CAMBIAR A len(BOUNDS)?

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
    colnames = self.params_names.split() # nuevo
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

    sns.set(context='paper',style='white',
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
                color = '#0880DE',
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

def plot_correl(self, chains=None, hdf5_file=None, burning=0.0):

    ##ndim = len(self.bounds)

    if hdf5_file:
        f = h5py.File(hdf5_file, 'r')
        index = f['INDEX'].value[0]
        conver_steps = f['CONVER_STEPS'].value[0]
        converge_time = f['ITER_LAST'].value[0]
        chains = f['CHAINS'].value[:,:,:index+1,:]
        nwalkers= f['NWALKERS'].value[0]
        f.close()

        burning = int(burning*index)
        last_it = int(converge_time / conver_steps)
        #print(burning, last_it)
        chains = chains[0,:,burning:last_it,:]
    # PARECE QUE AQUI FALTA CONSIDERAR index PARA HACER EL BURNING SOBRE chains
    nwalkers = chains.shape[0]
    steps = chains.shape[1]
    chains_2D = chains.reshape(nwalkers*steps,self.NPLA*7) # self.ndim CAMBIAR A len(BOUNDS)?

    df = pd.DataFrame(chains_2D)
    plt.figure(figsize=[16,6])
    sns.heatmap(df.corr(), vmin=-1, vmax=1, cmap='coolwarm',annot=True)

    return

def plot_monitor(hdf5_file):

    sns.set(context='notebook',style='white')

    f = h5py.File(hdf5_file, 'r')
    bestchi2 = f['BESTCHI2'].value
    betas = f['BETAS'].value
    acc = f['ACC_FRAC0'].value
    meanchi2 = f['MEANCHI2'].value
    tau = f['TAU_PROM0'].value
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
    axes[1,0].plot(meanchi2[:index+1])
    axes[1,0].plot(bestchi2[:index+1])
    axes[1,0].set_ylabel(r"$-\chi^2$")
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


def plot_convergence(syn, chains=None, names=None, nchunks_gr=10, thinning=10):
    
    if names is None:
        # Create a generic list of names
        names = [f"Dim{d}" for d in list(range(chains.shape[-1]))] 
    
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(20,10))
    
    symbols = {1:"o",2:"^",3:"x",4:"s",5:"P"}
    
    # ========== Gelman Rubin test
    GR = gelman_rubin(chains, nchunks_gr=nchunks_gr, thinning=thinning, names=names)

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

    axes[0].set_title(f"{syn.system_name}", fontsize=15)
    axes[0].axhline( 1.01, color='gray', alpha=0.8, ls="--")
    axes[0].axhline( 1, color='k', alpha = 0.5, ls='--')
    axes[0].set_xlabel("Start iteration", fontsize=15)
    axes[0].set_ylabel(r'$\hat{R}$', fontsize=15)
    axes[0].legend(loc='best')#bbox_to_anchor=(1.04,1), loc="bottom left")
    axes[0].grid(alpha=0.1)    
        
    # ========== Geweke test
    Z = geweke(syn,  chains=chains, names=names, burning=0)
    
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
    #fig.subplots_adjust(right=0.75) 
    
    return {"GR": GR, "Z": Z}


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