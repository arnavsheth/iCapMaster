% Code for an Example using Algorithm 2 
% from the chapter "Markov Chain Monte Carlo Methods in Corporate Finance", 
%   In P. Damien, P. Dellaportas, N. Polson, and D. Stephens (Eds.), 
%   MCMC and  Hierarchical Models. Oxford University Press.
%
% Copyright (C) 2011 by Arthur Korteweg
%
% Simulate a linear regression model, randomly dropping observations on the 
% dependent variable. Then re-estimate the model using:
% 1) OLS, full data (benchmark, assuming we could observe the missing data)
% 2) OLS, dropping the observations with missing y
% 3) OLS, filling in missing y with fitted values from regression
% 4) MCMC, dropping the observations with missing y
% 5) MCMC, filling in missing y (Algorithm 2)
%
% The model is:
%   y = x'*b + epsilon       epsilon ~ N(0, sige^2)

clear all
close all

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Initialization
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Simulation parameters (if using simulated data)
sim.N         = 50;         % number of datapoints to simulate
sim.seed      = 1;          % seed for random number generator
sim.true_beta = 1;          % true intercept and slope
sim.true_sige = 0.25;       % true standard deviation of error
sim.dropprob  = 0.5;        % probability of dropping a given datapoint (0 to 1)

% MCMC parameters
MCMC_options.G_burnin   = 1000;     % # burn-in draws (to be discarded)
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
y_full = y;      % full observed data (before dropping observations)    

% Randomly drop a fraction of the observations of y
II    = rand(sim.N,1) < sim.dropprob;   % indicator = 1 if observation dropped
y(II) = NaN;                            % denote dropped data by NaN's


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Estimation 1: Run OLS regression, full sample
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if exist ('y_full')
    disp('OLS: full data')
    OLS_full = regr (y_full, x)
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Estimation 2: Run OLS regression, dropping the observations with missing y
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp('OLS: drop missing data')
II = ~isnan(y);
OLS_drop = regr (y(II), x(II,:))    % NB: R^2, F, and p are meaningless here, 
                                    % because there is no intercept in the regression


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Estimation 3: Run OLS regression, filling in missing y with fitted
% values from regression
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp('OLS: fill in missing data from regression')
y_fill = y; y_fill(isnan(y_fill)) = x(isnan(y_fill),:)*OLS_drop.b;
OLS_fill = regr (y_fill, x)         % NB: R^2, F, and p are meaningless here, 
                                    % because there is no intercept in the regression


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Estimation 4: MCMC, dropping the missing data (Algorithm 1)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp('MCMC: drop missing data');

% set priors, see MCMC_regression.m for details
prior.a  = 2.1;     % NB: mean of Inverse Gamma = b / (a-1)   if a>1
prior.b  = 1;       %     variance = b^2 / [(a-1)^2 * (a-2)]  if a>2
prior.mu = zeros(size(x,2),1);
prior.A  = eye(size(x,2))/10000;

MCMC_drop_output = MCMC_regression (y(II), x(II,:), prior, MCMC_options);

disp('Posterior:');
disp(['Mean beta = ' num2str(mean(MCMC_drop_output.beta(MCMC_options.G_burnin+1:end, :)))]);
disp(['Std  beta = ' num2str(std(MCMC_drop_output.beta(MCMC_options.G_burnin+1:end, :)))]);
disp(' ');
disp(['Mean sigma = ' num2str(mean(sqrt(MCMC_drop_output.sige2(MCMC_options.G_burnin+1:end, :))))]);
disp(['Std  sigma = ' num2str(std(sqrt(MCMC_drop_output.sige2(MCMC_options.G_burnin+1:end, :))))]);
disp(' ');


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Estimation 5: MCMC, filling in missing y (Algorithm 2)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp('MCMC: fill in missing data from regression');

tic

MCMC_fill_output = MCMC_randommissing (y, x, prior, MCMC_options);

toc

disp('Posterior');
disp(['Mean beta = ' num2str(mean(MCMC_fill_output.beta(MCMC_options.G_burnin+1:end, :)))]);
disp(['Std  beta = ' num2str(std(MCMC_fill_output.beta(MCMC_options.G_burnin+1:end, :)))]);
disp(' ');
disp(['Mean sigma = ' num2str(mean(sqrt(MCMC_fill_output.sige2(MCMC_options.G_burnin+1:end, :))))]);
disp(['Std  sigma = ' num2str(std(sqrt(MCMC_fill_output.sige2(MCMC_options.G_burnin+1:end, :))))]);
disp(' ');


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Plot posterior distributions of beta and sigma
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
f = '%6.3f';
figure;
title('MCMC Posterior Histograms')       
for i = 1:size(x,2);     % plot the histogram of each coefficient's posterior distribution
    subplot(ceil((size(x,2)+1)/2),2,i); 
    histplot(MCMC_fill_output.beta(MCMC_options.G_burnin+1:end, i), 25, sim.true_beta(i)); 
    title(['\beta_' num2str(i-1)]);
 
end

% plot the histogram of the error standard deviation
sige = sqrt(MCMC_fill_output.sige2);
subplot(ceil((size(x,2)+1)/2),2,size(x,2)+1); 
histplot(sige(MCMC_options.G_burnin+1:end,1), 25, sim.true_sige);
title('\sigma');     

% Trace plots
figure;
title('MCMC Parameter trace plots')
G = 50;     % plot the first G cycles of the algorithm
for i = 1:size(x,2);
    subplot(ceil((size(x,2)+1)/2),2,i);    
    traceplot(MCMC_fill_output.beta(:,i), G, sim.true_beta(i));
    title(['\beta_' num2str(i-1)]);
end
subplot(ceil((size(x,2)+1)/2),2,size(x,2)+1); 
traceplot(sqrt(MCMC_fill_output.sige2), G, sim.true_sige);
title('\sigma');       
