=============================================================================
This is a set of Matlab routines to simulate data and re-estimate the models 
used in the book chapter "Markov Chain Monte Carlo Methods in Corporate 
Finance".

You are free to use and adapt this code for your own, non-commercial research.
I only ask that you cite the book chapter in your work:

Korteweg, A. (2013), Markov Chain Monte Carlo Methods in Corporate Finance, 
In P. Damien, P. Dellaportas, N. Polson, and D. Stephens (Eds.), Bayesian 
Theory and Applications. Oxford University Press.
=============================================================================


Description:
-----------------------------------------------------------------------------
The main files are Matlab routines that start with "Ex...". They simulate 
data and re-estimate the models of the book chapter. The numbers refer to 
the corresponding algorithms in the chapter. 

The examples call the Matlab routines that start with "MCMC_...", which 
contain the actual MCMC algorithms.



List of examples and associated files
-----------------------------------------------------------------------------
Ex1_regression.m		Simple linear regression example, uses 
   MCMC_regression.m			Algorithm 1 in the chapter
Ex2_randommissing.m		Regression with randomly missing dependent data
   MCMC_randommissing.m			Algorithm 2 in the chapter
Ex3_probit.m			Probit regression
   MCMC_probit.m			Algorithm 3 in the chapter
Ex4_Heckman.m			Heckman selection model
   MCMC_Heckman.m			Algorithm 4 in the chapter
Ex5_probit_randomeffects.m	Panel probit model with random effects
   MCMC_probit_randomeffects.m		Algorithm 5 in the chapter
Ex6_Heckman_randomeffects.m	Panel linear regression with random effects, 
				and Heckman-type selection
   MCMC_Heckman_randomeffects.m		Algorithm 6 in the chapter



Auxiliary programs (in alphabetical order):
-----------------------------------------------------------------------------
Bayesregr.m		Calculates the parameters of the posterior 
			distribution of a (Bayesian) linear regression
histplot.m		Plots a histogram with kernel approximation
randn_uppertrunc.m	Draws from an upper-truncated Normal distribution
regr.m			OLS regression
traceplot.m		Graphs a trace plot of the MCMC sampled values 
			across iterations
