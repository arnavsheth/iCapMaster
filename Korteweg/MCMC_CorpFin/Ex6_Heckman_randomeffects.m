% Code for an Example using Algorithm 6
% from the chapter "Markov Chain Monte Carlo Methods in Corporate Finance", 
%   In P. Damien, P. Dellaportas, N. Polson, and D. Stephens (Eds.), 
%   MCMC and  Hierarchical Models. Oxford University Press.
%
% Copyright (C) 2011 by Arthur Korteweg
%
% Simulate a random effects panel regression model with selection and 
% re-estimate using:
% 1) OLS, full data (benchmark, assuming we could observe the missing data)
% 2) OLS, ignoring the selection problem
% 3) MCMC, accounting for the selection problem (Algorithm 4)
%
% The model is:
%   y(it) = alpha(i) + x(it)'*beta + epsilon(it)    epsilon ~ N(0, sige^2)     observation eqn
%   P(D(it)=1|z) = Phi(theta(i) + z(it)'*gamma)     Phi(.) is N(0,1) cdf       selection eqn
%
%   where D = 1 if y is observed, and D = 0 otherwise
%
%   alpha and theta are independent random effects.
%   alpha(i) ~ N(mu, tau^2)
%   theta(i) ~ N(kappa, omega^2)
%
%   The correlation between the error terms in the observation and selection
%   equations is "rho".

clear all
close all

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Initialization 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Simulation parameters (if using simulated data)
sim.T          = 100;   % number of time periods to simulate for each firm
sim.N          = 100;   % number of firms to simulate
sim.seed       = 1;     % seed for random number generator
sim.true_beta  = 1;     % true intercept and slope
sim.true_gamma = 0.5; 	% true intercept and slope
sim.true_rho   = 0.5;   % correlation between errror term in obs eqn and selection eqn
sim.true_sige  = 1;     % standard deviation of the observation eqn error term
sim.true_mu    = 0;     % mean of random effect in obs eqn
sim.true_tau   = 0.5;   % stdev of random effect in obs eqn
sim.true_kappa = 0;     % mean of random effect in selection eqn
sim.true_omega = 0.25;  % stdev of random effect in selection eqn

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
z = randn(sim.N*sim.T, length(sim.true_gamma));
alpha = sim.true_mu + sim.true_tau * randn(sim.N,1);        % random effects in obs eqn
theta = sim.true_kappa + sim.true_omega * randn(sim.N,1);   % random effects in selection eqn
sim.true_sigeta = sqrt(1 - sim.true_omega^2);
e = mvnrnd([0 0], [sim.true_sige^2 sim.true_rho*sim.true_sige*sim.true_sigeta; sim.true_rho*sim.true_sige*sim.true_sigeta sim.true_sigeta^2], sim.N*sim.T);
firmid = kron(ones(sim.T,1), (1:sim.N)');

y_full = kron(ones(sim.T,1),alpha) + x * sim.true_beta + e(:,1);
w = kron(ones(sim.T,1),theta) + z * sim.true_gamma + e(:,2);
y = y_full;
y(w < 0) = NaN;     % denote dropped data by NaN's


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
disp('OLS: ignore selection problem by dropping missing data')
II = ~isnan(y);
OLS_drop = regr (y(II), x(II,:));


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Estimate MCMC random effects panel model with selection
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp('MCMC random effects panel model with selection');
% set priors, see MCMC_Heckman_randomeffects.m for details
prior.mu_beta   = zeros(size(x,2),1);
prior.A_beta    = eye(size(x,2))/10000;
prior.mu_gamma  = zeros(size(z,2),1);
prior.A_gamma   = eye(size(z,2))/10000;
prior.mu_delta  = 0;
prior.A_delta   = 1/10000;
prior.a         = 2.1;
prior.b         = 1;
% hyperpriors
prior.mu_mu     = 0;
prior.A_mu      = 1/10000;
prior.a_tau     = 2.1;
prior.b_tau     = 1;
prior.mu_kappa  = 0;
prior.A_kappa   = 1/10000;
prior.a_omega   = 2.1;
prior.b_omega   = 1;

tic

MCMC_output = MCMC_Heckman_randomeffects (y, x, z, firmid, prior, MCMC_options);

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
disp(['Mean mu = ' num2str(mean(MCMC_output.mu(MCMC_options.G_burnin+1:end, :)))]);
disp(['Std  mu = ' num2str(std(MCMC_output.mu(MCMC_options.G_burnin+1:end, :)))]);
disp(' ');
disp(['Mean tau = ' num2str(mean(sqrt(MCMC_output.tau2(MCMC_options.G_burnin+1:end, :))))]);
disp(['Std  tau = ' num2str(std(sqrt(MCMC_output.tau2(MCMC_options.G_burnin+1:end, :))))]);
disp(' ');
disp(['Mean kappa = ' num2str(mean(MCMC_output.kappa(MCMC_options.G_burnin+1:end, :)))]);
disp(['Std  kappa = ' num2str(std(MCMC_output.kappa(MCMC_options.G_burnin+1:end, :)))]);
disp(' ');
disp(['Mean omega = ' num2str(mean(sqrt(MCMC_output.omega2(MCMC_options.G_burnin+1:end, :))))]);
disp(['Std omega = ' num2str(std(sqrt(MCMC_output.omega2(MCMC_options.G_burnin+1:end, :))))]);
disp(' ');


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Plot posterior distributions of beta, gamma, sigma, rho, tau, and omega,
% and the histogram of the posterior mean random effects for each firm.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%figure;
% Histograms of the regression parameters
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

% Histograms of the random effects means and standard deviations
figure;
tau = sqrt(MCMC_output.tau2);
omega = sqrt(MCMC_output.omega2);
subplot(2,2,1); histplot(MCMC_output.mu(MCMC_options.G_burnin+1:end,1), 25, sim.true_mu); title('\mu');     
subplot(2,2,2); histplot(MCMC_output.kappa(MCMC_options.G_burnin+1:end,1), 25, sim.true_kappa); title('\kappa');     
subplot(2,2,3); histplot(tau(MCMC_options.G_burnin+1:end,1), 25, sim.true_tau); title('\tau');     
subplot(2,2,4); histplot(omega(MCMC_options.G_burnin+1:end,1), 25, sim.true_omega); title('\omega');     

% Trace plots of the regression parameters
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

% Trace plots of the random effects means and standard deviations
figure;
subplot(2,2,1);
traceplot(MCMC_output.mu, G, sim.true_mu);
title('\mu');   
subplot(2,2,2);
traceplot(MCMC_output.kappa, G, sim.true_kappa);
title('\kappa');   
subplot(2,2,3);
traceplot(sqrt(MCMC_output.tau2), G, sim.true_tau);
title('\tau');   
subplot(2,2,4);
traceplot(sqrt(MCMC_output.omega2), G, sim.true_omega);
title('\omega');   

% Histogram of the posterior mean random effects for each firm.
figure
subplot(2,1,1); histplot(mean(MCMC_output.alpha(MCMC_options.G_burnin+1:end,:)), 25); 
title('Histogram of posterior mean \alpha_i');    
subplot(2,1,2); histplot(mean(MCMC_output.theta(MCMC_options.G_burnin+1:end,:)), 25); 
title('Histogram of posterior mean \theta_i');    