% Code for an Example using Algorithm 5
% from the chapter "Markov Chain Monte Carlo Methods in Corporate Finance", 
%   In P. Damien, P. Dellaportas, N. Polson, and D. Stephens (Eds.), 
%   MCMC and  Hierarchical Models. Oxford University Press.
%
% Copyright (C) 2011 by Arthur Korteweg
%
% Simulate a random effects panel probit model and re-estimate using MCMC
%
% The model is:
%   P(y(it)=1|x) = Phi(alpha(i) + x(it)'*beta)     where Phi(.) is the standard Normal cdf
%
%   and alpha(i) ~ N(0, tau^2)

clear all
close all

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Initialization 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Simulation parameters (if using simulated data)
sim.T           = 100;  % number of time periods to simulate for each firm
sim.N           = 100;  % number of firms to simulate
sim.seed        = 1;    % seed for random number generator
sim.true_beta   = 1;    % true slope
sim.true_tau    = 0.5;  % stdev of random effect (NB: <1, theoretically)

% MCMC parameters
MCMC_options.G_burnin   = 1000;     % # burn-in draws (to be discarded)
MCMC_options.G_samples  = 1000;     % # draws to sample from posterior distribution
MCMC_options.seed       = 1;        % random number generator seed (convenient for debugging)
                             

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Simulate data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Initialize random number generator
s = RandStream('mcg16807', 'Seed', sim.seed);
RandStream.setDefaultStream(s);

% generate data
x = randn(sim.N*sim.T, length(sim.true_beta));
alpha = sim.true_tau * randn(sim.N,1);        % random effects
firmid = kron(ones(sim.T,1), (1:sim.N)');    % identifier for each firm

y = (kron(ones(sim.T,1),alpha) + x * sim.true_beta + randn(sim.N*sim.T,1) >= 0);    


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Estimate random effects probit model by MCMC
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp('MCMC random effects model');
% set priors, see MCMC_probit_randomeffects.m for details
prior.mu = zeros(size(x,2),1);
prior.A  = eye(size(x,2))/10000;
prior.a = 2.1;
prior.b = 1;

tic

MCMC_output = MCMC_probit_randomeffects (y, x, firmid, prior, MCMC_options);

toc

disp('Posterior:');
disp(['Mean beta = ' num2str(mean(MCMC_output.beta(MCMC_options.G_burnin+1:end, :)))]);
disp(['Std  beta = ' num2str(std(MCMC_output.beta(MCMC_options.G_burnin+1:end, :)))]);
disp(' ');
disp(['Mean tau = ' num2str(mean(sqrt(MCMC_output.tau2(MCMC_options.G_burnin+1:end, :))))]);
disp(['Std  tau = ' num2str(std(sqrt(MCMC_output.tau2(MCMC_options.G_burnin+1:end, :))))]);
disp(' ');


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Plot posterior distributions of beta and tau, 
% and the histogram of the posterior mean random effect for each firm.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Histograms
figure;
title('MCMC Posterior Histograms')       
for i = 1:size(x,2);
    subplot(ceil((size(x,2)+1)/2),2,i); 
    histplot(MCMC_output.beta(MCMC_options.G_burnin+1:end, i), 25, sim.true_beta(i)); 
    title(['\beta_' num2str(i-1)]);
end
tau = sqrt(MCMC_output.tau2);
subplot(ceil((size(x,2)+1)/2),2,size(x,2)+1); histplot(tau(MCMC_options.G_burnin+1:end,1), 25, sim.true_tau); title('\tau');     

% Trace plots
figure;
title('MCMC Parameter trace plots')
G = MCMC_options.G_burnin;    % Show the first G draws of the chain
for i = 1:size(x,2);
    subplot(ceil((size(x,2)+1)/2),2,i);
    traceplot(MCMC_output.beta(:,i), G, sim.true_beta(i));
    title(['\beta_' num2str(i-1)]);
end
subplot(ceil((size(x,2)+1)/2),2,size(x,2)+1); 
traceplot(sqrt(MCMC_output.tau2), G, sim.true_tau);
title('\tau');     

% Histogram of the posterior mean random effect for each firm.
figure
title('Histogram of posterior mean \alpha_i');     
histplot(mean(MCMC_output.alpha(MCMC_options.G_burnin+1:end,:)), 25); 