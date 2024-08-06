% Code for an Example using Algorithm 3
% from the chapter "Markov Chain Monte Carlo Methods in Corporate Finance", 
%   In P. Damien, P. Dellaportas, N. Polson, and D. Stephens (Eds.), 
%   MCMC and  Hierarchical Models. Oxford University Press.
%
% Copyright (C) 2011 by Arthur Korteweg
%
% Simulate a Probit model and re-estimate the model using MCMC
%
% The model is:
%   P(y=1|x) = Phi(x'*beta)     where Phi(.) is the standard Normal cdf

clear all
close all

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Initialization
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Simulation parameters
sim.N         = 1000;   % number of datapoints to simulate
sim.seed      = 1;      % seed for random number generator
sim.true_beta = [0;1];  % true intercept and slope

% MCMC parameters
MCMC_options.G_burnin   = 1000;     % # burn-in draws (to be discarded)
MCMC_options.G_samples  = 10000;    % # draws to sample from posterior distribution
MCMC_options.seed       = 1;        % random number generator seed (convenient for debugging)
                             

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Simulate data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
% Initialize random number generator
s = RandStream('mcg16807', 'Seed', sim.seed);
RandStream.setDefaultStream(s);

% generate data
x = [ones(sim.N,1) randn(sim.N, length(sim.true_beta)-1)];
y = (x * sim.true_beta + randn(sim.N, 1) >= 0);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Estimate Probit model by MCMC
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp('MCMC Probit model');

% set priors, see MCMC_probit.m for details
prior.mu = zeros(size(x,2),1);
prior.A  = eye(size(x,2))/10000;

tic

MCMC_output = MCMC_probit (y, x, prior, MCMC_options);

toc

disp('Posterior:');
disp(['Mean beta = ' num2str(mean(MCMC_output.beta(MCMC_options.G_burnin+1:end, :)))]);
disp(['Std  beta = ' num2str(std(MCMC_output.beta(MCMC_options.G_burnin+1:end, :)))]);
disp(' ');


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Plot posterior distribution of beta
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Histograms
figure;
title('MCMC Posterior Histograms')       
for i = 1:size(x,2);
    subplot(ceil(size(x,2)/2),2,i); 
    histplot(MCMC_output.beta(MCMC_options.G_burnin+1:end, i), 25, sim.true_beta(i)); 
    title(['\beta_' num2str(i-1)]);
end

% Trace plots
figure;
title('MCMC Parameter trace plots')
G = MCMC_options.G_burnin;    % Show the first G draws of the chain
for i = 1:size(x,2);
    subplot(ceil(size(x,2)/2),2,i);    
    traceplot(MCMC_output.beta(:,i), G, sim.true_beta(i));
    title(['\beta_' num2str(i-1)]);
end
