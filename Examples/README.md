
# DESCRIPTION

This directory contains many examples for using nauyaca along with many other python tools. We recommend see the functionalities of the modules at Documentation before running the examples, so the user becomes familiar with the main modules and functions.

All input files (as planet ephemeris) are stored in the folder named 'inputs'. For many examples the same transit ephemeris are used. We recommend running these examples in the same order as listed here. These examples show many extra features hosted in nauyaca. 

In order to run these examples in a permissible wall-clock time, we suggest using a multi-core machine. Otherwise, these examples can take longer times to complete. The html directory contains these examples in html format.


# MAIN EXAMPLES

**simple_fit**: Perform the fit of a three-planet system using the optimization and MCMC modules. Generate a pair of built-in figures to visualize the results. Unlike the example shown in the documentation, this example shows the minimum code needed to perform yout fit.


**customized_fit**: Perform the fit of a two-planet system using a initial walker population and prior given by the user. Many other features in nauyaca are introduced to customize the fitting process. Plot helpful figures to assess for convergence.



# OTHER EXAMPLES:

For the following examples, run first the two examples above since these results will be used.


**initial_walkers**: An example of how to chose the best built-in initialization strategy from the optimization results. For this example, simple_fit results are needed.


**restarting_mcmc**: The mcmc did not converge? Probably more iterations are needed. Restart an mcmc run from previous results. For this example, customized_fit results are needed.


**customized_figures**: Change figure options to adapted to your needs.  Plot the built-in figures to include in your research. For this example, results from simple_fit and customized_fit are needed.


**miscellany**: Many other functionalities adapted in nauyaca. Test planetary proposals to build transit ephemeris. Get the chi square and log-likelihood of the proposals. Common errors and warnings are shown in order to correct them. Also, change default settings in nauyaca. For this example, simple_fit results are needed.




Modify these scripts to adapt to your own problem. More examples will be available soon!

