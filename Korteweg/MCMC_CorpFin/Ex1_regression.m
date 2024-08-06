% Code for an Example using Algorithm 1 
% from the chapter "Markov Chain Monte Carlo Methods in Corporate Finance", 
%   In P. Damien, P. Dellaportas, N. Polson, and D. Stephens (Eds.), 
%   MCMC and  Hierarchical Models. Oxford University Press.
%
% Copyright (C) 2011 by Arthur Korteweg
%
% Simulate and re-estimate a linear regression model using:
% 1) OLS
% 2) Analytical Bayesian regression
% 3) MCMC
%
% The model is:
%   y = x'*beta + epsilon       epsilon ~ N(0, sige^2)

clear all
close all

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Initialization
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Simulation parameters (if using simulated data)
sim.N         = 100;        % number of datapoints to simulate
sim.seed      = 1;          % seed for random number generator
sim.true_beta = 1;          % true intercept and slope
sim.true_sige = 0.25;       % true standard deviation of error

% MCMC parameters
MCMC_options.G_burnin   = 0;        % # burn-in draws (to be discarded)
MCMC_options.G_samples  = 25000;    % # draws to sample from posterior distribution
MCMC_options.seed       = 1;        % random number generator seed (convenient for debugging)
                             

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Simulate data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
% Initialize random number generator
s = RandStream('mcg16807', 'Seed', sim.seed);
RandStream.setDefaultStream(s);

% generate data
x = randn(sim.N, length(sim.true_beta));
y = zeros(sim.N, 1); 
y = x*sim.true_beta + sim.true_sige * randn(sim.N,1);   


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Estimation 1: Run OLS regression
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp('OLS: full data')
OLS = regr (y, x)       % NB: R^2, F, and p are meaningless here, 
                        % because there is no intercept in the regression


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Estimation 2: Bayesian regression (analytical)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp('Analytical Bayesian regression')

% set priors, see Bayesregr.m for details
prior.a  = 2.1;     % NB: mean of Inverse Gamma = b / (a-1)   if a>1
prior.b  = 1;       %     variance = b^2 / [(a-1)^2 * (a-2)]  if a>2
prior.mu = zeros(size(x,2),1);
prior.A  = eye(size(x,2))/10000;

post_analytic = Bayesregr (y, x, prior);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Estimation 3: MCMC Algorithm 1
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp('MCMC');

tic

MCMC_output = MCMC_regression (y, x, prior, MCMC_options);

toc

disp('Posterior:');
disp(['Mean beta = ' num2str(mean(MCMC_output.beta(MCMC_options.G_burnin+1:end, :)))]);
disp(['Std  beta = ' num2str(std(MCMC_output.beta(MCMC_options.G_burnin+1:end, :)))]);
disp(' ');
disp(['Mean sigma = ' num2str(mean(sqrt(MCMC_output.sige2(MCMC_options.G_burnin+1:end, :))))]);
disp(['Std  sigma = ' num2str(std(sqrt(MCMC_output.sige2(MCMC_options.G_burnin+1:end, :))))]);
disp(' ');


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Plot posterior distributions of beta and sigma
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
f = '%6.3f';
figure;
title('MCMC Posterior Histograms')       
for i = 1:size(x,2)     % plot the histogram of each coefficient's posterior distribution
    subplot(1,size(x,2)+1,i);
    histplot(MCMC_output.beta(MCMC_options.G_burnin+1:end, i), 25, sim.true_beta(i)); 
    hold on;
    postmean = mean(MCMC_output.beta(MCMC_options.G_burnin+1:end, i));
	annotation('textbox',[0.15 0.85 0.1 0.1],'String',{['True \beta = ' num2str(sim.true_beta(i),f)] ['Posterior mean = ' num2str(postmean,f)] ['OLS = ' num2str(OLS.b(i),f)]});
    title(['Posterior distribution of \beta_' num2str(i-1)]);
end

% plot the histogram of the error standard deviation
subplot(1,size(x,2)+1,size(x,2)+1);
sige = sqrt(MCMC_output.sige2);
histplot(sige(MCMC_options.G_burnin+1:end,1), 25, sim.true_sige);
hold on;
postmean = mean(sige(MCMC_options.G_burnin+1:end, i));
annotation('textbox',[0.65 0.85 0.1 0.1],'String',{['True \sigma = ' num2str(sim.true_sige(i),f)] ['Posterior mean = ' num2str(postmean,f)] ['OLS = ' num2str(OLS.stde,f)]});
title('Posterior distribution of \sigma');     
