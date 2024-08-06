=============================================================================
This is a set of routines to simulate data and re-estimate the model used 
in the paper "Risk and Return of Venture Capital-Backed Entrepreneurial 
Companies". 

You are free to use and adapt this code for your own, non-commercial research, 
provided you quote the original article:

Korteweg, A. and M. Sorensen (2010), Risk and Return of Venture Capital-Backed 
Entrepreneurial Companies, Review of Financial Studies 23 (10), p.3738-3772.
=============================================================================


See MCMC_algorithm.pdf for a detailed explanation of the algorithm used to 
estimate the model.


Instructions to running the code:

First run "Simdata.m" to simulate data from the model.

To re-estimate the model in Matlab, run "MCMC.m"

To re-estimate using c++:
i) run "MCMC.exe" (NB: this program was compiled under Linux. To run it under
		Windows we suggest using Cygwin or another UNIX/Linux environment)
ii) run "MCMCout_sum.m" to summarize and graph the output

NB: The source code to the c++ algorithm is provided in MCMC.cpp, which also
    contains compilation instructions.



Alphabetical list of files and brief description:
histplot.m		Function to plot a histogram with kernel approximation
MCMC.cpp		MCMC estimation algorithm in C++
MCMC.exe		Compiled version of MCMC.cpp
MCMC.m			MCMC estimation algorithm in matlab
MCMC_algorithm.pdf	Detailed description of the MCMC algorithm
MCMCout_b.dat		Parameter draws from MCMC.cpp
MCMCout_gamma.dat	Parameter draws from MCMC.cpp
MCMCout_info.dat	Number of iterations used in MCMC.cpp
MCMCout_sige2.dat	Parameter draws from MCMC.cpp
MCMCout_sum.m		Summarizes the C++ output
randn_lowertrunc.m	Draw from a lower-truncated Normal (used by MCMC.m)
randn_uppertrunc.m	Draw from an upper-truncated Normal (used by MCMC.m)
Simdata.m		Simulates data from the model
Simdata.mat		Saved simulated data for use in MCMC.m
Simdata_info.dat	Saved simulated data for MCMC.cpp
Simdata_logVobs.dat	Saved simulated data for MCMC.cpp
Simdata_rmrf.dat	Saved simulated data for MCMC.cpp