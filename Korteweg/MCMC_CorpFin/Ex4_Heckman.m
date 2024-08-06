% Code for an Example using Algorithm 4
% from the chapter "Markov Chain Monte Carlo Methods in Corporate Finance", 
%   In P. Damien, P. Dellaportas, N. Polson, and D. Stephens (Eds.), 
%   MCMC and  Hierarchical Models. Oxford University Press.
%
% Copyright (C) 2011 by Arthur Korteweg
%
% Simulate and re-estimate a Heckman selection model using:
% 1) OLS, full data (benchmark, assuming we could observe the missing data)
% 2) OLS, ignoring the selection problem
% 3) MCMC, accounting for the selection problem (Algorithm 4)
%
% The model is:
%   y = x'*beta + epsilon        epsilon ~ N(0, sige^2)     observation eqn
%   P(D=1|z) = Phi(z'*gamma)     Phi(.) is N(0,1) cdf       selection eqn
%
%   where D = 1 if y is observed, and D = 0 otherwise
%
%   The correlation between the error terms in the observation and selection
%   equations is "rho".

clear all
close all

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Initialization
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Simulation parameters (if using simulated data)
sim.N          = 1000;      % number of datapoints to simulate
sim.seed       = 1;         % seed for random number generator
sim.true_beta  = [0; 1];    % true intercept and slope in observation eqn
sim.true_gamma = [.5; .5];  % true intercept and slope in selection eqn
sim.true_rho   = 0.5;       % correlation between errror term in obs eqn and selection eqn
sim.true_sige  = 1;         % standard deviation of the observation eqn error term

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
z = [ones(sim.N,1) randn(sim.N, length(sim.true_gamma)-1)];
e = mvnrnd([0 0], [sim.true_sige^2 sim.true_rho*sim.true_sige; sim.true_rho*sim.true_sige 1], sim.N);
y_full = x * sim.true_beta + e(:,1);
w = z * sim.true_gamma + e(:,2);
y = y_full;
y(w < 0) = NaN;       % denote dropped data by NaN's


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Estimate OLS regression of y on x, data (benchmark, assuming we could 
% observe the missing data)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if exist ('y_full')
    disp('OLS: full data')
    OLS_full = regr (y_full, x);
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Estimate OLS regression of y on x, ignoring selection problem
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp('OLS: ignore selection problem by dropping the missing data')
II = ~isnan(y);
OLS_drop = regr (y(II), x(II,:));


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Estimate Heckman model by MCMC
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp('MCMC Heckman model');

% set priors, see MCMC_Heckman.m for details
prior.mu_beta   = zeros(size(x,2),1);
prior.A_beta    = eye(size(x,2))/10000;
prior.mu_gamma  = zeros(size(z,2),1);
prior.A_gamma   = eye(size(z,2))/10000;
prior.mu_delta  = 0;
prior.A_delta   = 1/100;
prior.a         = 2.1;
prior.b         = 1;

tic

MCMC_output = MCMC_Heckman (y, x, z, prior, MCMC_options);

toc

disp('Posterior:');
disp(['Mean beta = ' num2str(mean(MCMC_output.beta(MCMC_options.G_burnin+1:end, :)))]);
disp(['Std  beta = ' num2str(std(MCMC_output.beta(MCMC_options.G_burnin+1:end, :)))]);
disp(' ');
disp(['Mean gamma = ' num2str(mean(MCMC_output.gamma(MCMC_options.G_burnin+1:end, :)))]);
disp(['Std  gamma = ' num2str(std(MCMC_output.gamma(MCMC_options.G_burnin+1:end, :)))]);
disp(' ');
disp(['Mean sigma = ' num2str(mean(sqrt(MCMC_output.sige2(MCMC_options.G_burnin+1:end, :))))]);
disp(['Std  sigma = ' num2str(std(sqrt(MCMC_output.sige2(MCMC_options.G_burnin+1:end, :))))]);
disp(' ');
disp(['Mean rho = ' num2str(mean(MCMC_output.rho(MCMC_options.G_burnin+1:end, :)))]);
disp(['Std  rho = ' num2str(std(MCMC_output.rho(MCMC_options.G_burnin+1:end, :)))]);
disp(' ');


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Plot posterior distributions of beta, gamma, sigma and rho
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Histograms
figure;
title('MCMC Posterior Histograms')       
for i = 1:size(x,2);
    subplot(ceil((size(x,2)+size(z,2))/2)+1,2,i); 
    histplot(MCMC_output.beta(MCMC_options.G_burnin+1:end, i), 25, sim.true_beta(i)); 
    title(['\beta_' num2str(i-1)]);
end
for i = 1:size(z,2);
    subplot(ceil((size(x,2)+size(z,2))/2)+1,2,size(x,2)+i); 
    histplot(MCMC_output.gamma(MCMC_options.G_burnin+1:end, i), 25, sim.true_gamma(i)); 
    title(['\gamma_' num2str(i-1)]);
end
sige = sqrt(MCMC_output.sige2);
subplot(ceil((size(x,2)+size(z,2))/2)+1,2,size(x,2)+size(z,2)+1); histplot(sige(MCMC_options.G_burnin+1:end,1), 25, sim.true_sige); title('\sigma');     
subplot(ceil((size(x,2)+size(z,2))/2)+1,2,size(x,2)+size(z,2)+2); histplot(MCMC_output.rho(MCMC_options.G_burnin+1:end,1), 25, sim.true_rho); title('\rho');     

% Trace plots
figure;
title('MCMC Parameter trace plots')
G = MCMC_options.G_burnin;    % Show the first G draws of the chain
for i = 1:size(x,2);
    subplot(ceil((size(x,2)+size(z,2))/2)+1,2,i);
    traceplot(MCMC_output.beta(:,i), G, sim.true_beta(i));
    title(['\beta_' num2str(i-1)]);
end
for i = 1:size(z,2);
    subplot(ceil((size(x,2)+size(z,2))/2)+1,2,size(x,2)+i); 
    traceplot(MCMC_output.gamma(:,i), G, sim.true_gamma(i));
    title(['\gamma_' num2str(i-1)]);
end
sige = sqrt(MCMC_output.sige2);
subplot(ceil((size(x,2)+size(z,2))/2)+1,2,size(x,2)+size(z,2)+1); 
traceplot(sqrt(MCMC_output.sige2), G, sim.true_sige);
title('\sigma');     
subplot(ceil((size(x,2)+size(z,2))/2)+1,2,size(x,2)+size(z,2)+2); 
traceplot(MCMC_output.rho, G, sim.true_rho);
title('\rho');     