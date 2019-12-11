#import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from utils import *
from dataclasses import dataclass
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import corner 
import h5py
import pandas as pd
import numpy as np

"""This is a modificable script.
Plots class is ran outside the classes that make the TTVs operations
"""


colors = {0:"red", 1:"olive", 2:"skyblue", 3:"gold", 4:"teal",
          5:"orange", 6:"purple"}

labels = {0:r"$\mathrm{Mass}$" ,
           1:r"$\mathrm{Period}$",
           2:r"$\mathrm{ecc}$",
           3:r"$\mathrm{Inc}$",
           4:r"$\mathrm{\omega}$",
           5:r"$\mathrm{M}$",
           6:r"$\mathrm{\Omega}$"}

units = {0:r"$\mathrm{[M_{\oplus}]}$",
         1:r"$\mathrm{[days]}$",
         2:r"$\mathrm{ }$",
         3:r"$\mathrm{[deg]}$",
         4:r"$\mathrm{[deg]}$",
         5:r"$\mathrm{[deg]}$",
         6:r"$\mathrm{[deg]}$"}


def plot_TTVs(self, flat_params):
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
    sns.set(context='paper')
    #sns.axes_style("darkgrid")
    #sns.set_style("darkgrid", {"axes.facecolor": ".9"})
    
    nplots = len(self.TTVs)

    _, axes = plt.subplots(figsize=(8,8), nrows=nplots, ncols=1, sharex=True)
    sns.despine()
    
    index = 0
    
    for pids, _ in self.planets_IDs.items():

        if hasattr(self.planets[pids], "ttvs_data"):
                
            # Read observed TTVs of current planet
            ttvs_dict = {k:self.TTVs[pids][k] for k 
                                    in sorted(self.TTVs[pids].keys())}
            errors = np.array([[v[1]*mins, v[2]*mins] for k,v 
                                    in ttvs_dict.items()]).T
            
            # Make a model O-C given by the transits
            ttvs_dict = {k:v[0] for k,v in ttvs_dict.items()}                
            x_obs, y_obs, model_obs = calculate_model(ttvs_dict)

            #residuals_1 = {x:y for x,y in zip(ttvs_dict.keys(), )} #(y_obs-model_obs.predict(x_obs))*mins

            # Plot observed TTVs
            sns.scatterplot(y_obs, (y_obs-model_obs.predict(x_obs))*mins, 
                            marker="o", ax=axes[index], color='k',
                            label=f'{pids}')
            axes[index].errorbar(y_obs, (y_obs-model_obs.predict(x_obs))*mins, 
                            yerr=errors, color='k', fmt='o', zorder=-1)
            

            # Make space for residuals
            divider = make_axes_locatable(axes[index])
            sub_ax = divider.append_axes("bottom", size="30%", pad=0.1)

            for solution in  flat_params:
                # Plot the best solutions given in flat_params
                SP = run_TTVFast(solution, mstar=self.mstar, 
                    NPLA=self.NPLA, Ftime=self.sim_interval )
                EPOCHS = calculate_epochs(SP, self)
                
                # Whitout model
                x_cal, y_cal, _ = calculate_model(EPOCHS[pids])
                
                axes[index].plot(y_cal, 
                            (y_cal-model_obs.predict(x_cal))*mins, '-+' )
                axes[index].set_ylabel("O-C [min]")
                axes[index].set_xlim(self.T0JD/2., self.Ftime+self.T0JD/2.)
                
                # TODO: Remove thicks in all figures except the last!!
                
                # Plot residuals
                residuals_2 = {x:y for x,y in zip(list(x_cal.flatten()), list(y_cal))} #(y_cal-model_obs.predict(x_cal))*mins
                #residuals = [ r1-r2 for r1, r2 in 
                #                zip(residuals_1, residuals_2)]
                residuals = [(ttvs_dict[r]-residuals_2[r])*mins for r in sorted(ttvs_dict.keys()) ]
                sub_ax.scatter(y_obs, residuals, s=5)
                sub_ax.set_xlim(self.T0JD/2., self.Ftime+self.T0JD/2.)
                
            index += 1

        else:
            print("No ttvs have been provided for planet {}".format(pids))
    
    plt.xlabel("Time [days]")
    
    return


def plot_hist(self, chains=None, hdf5_file=None, burning=0.0):
    """Make histograms of the every planetary parameter
    
    Arguments:
        chains {array} -- An array with shape (nwalkers, niters, nparams)
    """
    
    assert(0.0 <= burning <= 1.0), f"burning must be between 0 and 1!"

    if hdf5_file:
        # Extract chains from hdf5 file
        f = h5py.File(hdf5_file, 'r')
        index = f['INDEX'].value[0]
        chains = f['CHAINS'].value[:,:,:index+1,:]
        f.close()

        burning = int(burning*index)
        #last_it = int(converge_time / conver_steps)
        chains = chains[0,:,burning:index+1,:]

    sns.set(context='paper')
    _, axes = plt.subplots(nrows=self.NPLA, ncols=7, 
                            figsize=(16,3*self.NPLA))

    dim = 0
    for n in range(self.NPLA):
        for p in range(7):
            param_idx = (n*7) + p

            # Write labels
            if n == self.NPLA-1:
                axes[n, p].set_xlabel(labels[p]+" "+units[p], labelpad=10)
            if p == 0:
                planet_ID = [k for k,v in self.planets_IDs.items() if v==n]
                axes[n, p].set_ylabel(planet_ID[0], )

            # Plot histograms
            if param_idx not in list(self.constant_params.keys()):
                parameter = chains[:,:,dim].flatten()
                sns.distplot(parameter, kde=False, hist=True, 
                            color=colors[p], ax=axes[n, p])
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

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.15, hspace=0.3)
    
    return


def plot_chains(self, chains=None, hdf5_file=None, plot_means=True):
    """[summary]
    
    Keyword Arguments:
        chains {[type]} -- [description] (default: {None})
        hdf5_file {[type]} -- [description] (default: {None})
        plot_means {bool} -- If True, then plot the mean of the chains at each
                    iteration for all the dimensions (default: {True})
    """
    xlabel = 'Iteration / conver_steps '
    if hdf5_file:
        # Extract chains from hdf5 file
        f = h5py.File(hdf5_file, 'r')
        index = f['INDEX'].value[0]
        chains = f['CHAINS'].value[:,:,:index+1,:]
        conver_steps = f['CONVER_STEPS'].value[0]
        f.close()

        chains = chains[0,:,:index+1,:]
        xlabel = f'Iteration / {conver_steps} '

    sns.set(context='paper')
    _, axes = plt.subplots(nrows=len(self.bounds), ncols=1, 
                            figsize=(15, 10*self.NPLA), sharex=True)

    dim = 0
    for pla in range(self.NPLA):
        for param in range(7):
            param_idx = (pla*7) + param

            if param_idx not in list(self.constant_params.keys()):

                axes[dim].plot( chains[:,:,dim].T, 
                        color=colors[param], alpha=0.2)

                axes[dim].set_ylabel(labels[param]+str(pla+1)+ "\n" + 
                                            units[param], labelpad=10)
                dim += 1
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

    ndim = len(self.bounds) #npla*7
    #colnames = self.params_names.split()

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
    chains_2D = chains.reshape(nwalkers*steps,ndim) 

    """
    # Rename the columns of the data frame
    colnames = []
    for p in range(npla): 
        for _,v in labels.items(): 
            colnames.append(v+str(p+1))
    """
    colnames = []
    for pla in range(self.NPLA):
        for param in range(7):
            param_idx = (pla*7) + param

            if param_idx not in list(self.constant_params.keys()):
                colnames.append("\n" + labels[param]+str(pla+1) + "\n" )


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

    sns.set(context='paper', style="darkgrid", 
            font_scale=0.5,rc={"lines.linewidth": 0.5})

    corner.corner(df, quantiles=[0.16, 0.5, 0.84], 
                show_titles=True, 
                title_kwargs={"fontsize": 8}, title_fmt='.3f',
                label_kwargs={"fontsize": 8}, #"labelpad": 20},
                plot_contours=True, plot_datapoints=False, 
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

def plot_monitor(hdf5_file):

    sns.set(context='notebook')
    sns.set_style("white")

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


def calculate_model(ttvs_dict):
    X, Y = [], []
    for k, v in ttvs_dict.items():
        X.append(k)
        Y.append(v)
    
    X = np.array(X).reshape((-1, 1))
    Y = np.array(Y)
    
    model = LinearRegression().fit(X, Y)
    #b, m = model.intercept_, model.coef_

    return X, Y, model