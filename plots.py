#import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from utils import *
from dataclasses import dataclass
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


colors = {0:"red", 1:"olive", 2:"skyblue", 3:"gold", 4:"teal",
          5:"orange", 6:"purple"}

xlabels = {0:r"$\mathrm{Mass\ [M_{\oplus}]}$" ,
           1:r"$\mathrm{Period\ [days]}$",
           2:r"$\mathrm{e\cos\omega}$",
           3:r"$\mathrm{Inclination\ [deg]}$",
           4:r"$\mathrm{e\sin\omega}$",
           5:r"$\mathrm{Mean\ anomaly\ [deg]}$",
           6:r"$\mathrm{\Omega\ [deg]}$"}

@dataclass
class Plots:

    def plot_TTVs(self, chi2_params):
        mins = 1440.

        #sns.set_style("darkgrid")
        sns.set(context='paper')
        #sns.axes_style("darkgrid")
        #sns.set_style("darkgrid", {"axes.facecolor": ".9"})
        f, axes = plt.subplots(figsize=(8,8), nrows=len(self.TTVs), ncols=1, sharex=True)
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
                x_obs, y_obs, model_obs = self.calculate_model(ttvs_dict)
                residuals_1 = (y_obs-model_obs.predict(x_obs))*mins

                # Plot observed TTVs
                sns.scatterplot(y_obs, (y_obs-model_obs.predict(x_obs))*mins, 
                                marker="o", ax=axes[index], color='k',
                                label=f'{pids}')
                axes[index].errorbar(y_obs, (y_obs-model_obs.predict(x_obs))*mins, 
                                yerr=errors, color='k', fmt='o', zorder=-1)
                

                if chi2_params:
                    # Plot JUST the best solution in chi2_params
                    chi2_params.sort(key=lambda j: j[0])
                    best_sol = chi2_params[0][1]
                    
                    SP = run_TTVFast(best_sol, mstar=self.mstar, 
                        NPLA=self.NPLA, Ftime=self.Ftime)
                    EPOCHS = calculate_epochs(SP, self)
                    
                    # No model
                    x_cal, y_cal, _ = self.calculate_model(EPOCHS[pids])
                    
                    #sns.lineplot(y_cal, (y_cal-model.predict(x_cal))*mins, ax=axes[index])
                    axes[index].plot(y_cal, 
                                (y_cal-model_obs.predict(x_cal))*mins )
                    axes[index].set_ylabel("O-C [min]")
                    axes[index].set_xlim(0, self.Ftime)
                    
                    
                    # Plot residuals
                    divider = make_axes_locatable(axes[index])
                    sub_ax = divider.append_axes("bottom", size="30%", pad=0.1)
                    
                    # TODO: Remove thicks in all figures except the last!!

                    residuals_2 = (y_cal-model_obs.predict(x_cal))*mins
                    residuals = [ r1-r2 for r1, r2 in 
                                    zip(residuals_1, residuals_2)]
                    sub_ax.scatter(y_obs, residuals, s=5)
                    sub_ax.set_xlim(0, self.Ftime+1)
                    
                index += 1

            else:
                print("No ttvs have been provided for planet {}".format(pids))
        
        plt.xlabel("Time [days]")
        #plt.ylabel("O-C [min]")


    @staticmethod
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

    def plot_hist(self, chains):
        sns.set(context='paper')
        f, axes = plt.subplots(nrows=self.NPLA, ncols=7, 
                               figsize=(16,3*self.NPLA))

        for n in range(self.NPLA):
            for p in range(7):
                param_idx = (n*7) + p
                
                if n == self.NPLA-1:
                    axes[n, p].set_xlabel(xlabels[p], labelpad=10)
                if p == 0:
                    planet_ID = [k for k,v in self.planets_IDs.items() if v==n]
                    axes[n, p].set_ylabel(planet_ID[0], )

                
                parameter = chains[:,:,param_idx].flatten()
                sns.distplot(parameter, kde=False, hist=True, 
                            color=colors[p], ax=axes[n, p])
                low, med, up = np.percentile(parameter, [16,50,84])
                tit = r"$\mathrm{ %s ^{+%s}_{-%s} }$" % (round(med,2),
                                              round(med-low,3),
                                              round(up-med,3))
                axes[n, p].set_title(tit)
                axes[n, p].set_yticks([])
                
        plt.tight_layout()
        plt.subplots_adjust(wspace=0.15, hspace=0.3)
        return