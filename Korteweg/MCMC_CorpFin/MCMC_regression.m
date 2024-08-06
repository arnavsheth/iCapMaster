function output = MCMC_regression (y, x, prior, options)
% MCMC regression of y on x.
%
% Copyright (C) 2011 by Arthur Korteweg
%
% This is Algorithm 1 in the book chapter
% "Markov Chain Monte Carlo Methods in Corporate Finance", 
%   In P. Damien, P. Dellaportas, N. Polson, and D. Stephens (Eds.), 
%   MCMC and  Hierarchical Models. Oxford University Press.
%
%
% The model is:
%   y = x'*beta + epsilon       epsilon ~ N(0, sige^2)
%
%   y           Nx1 vector containing the dependent variable
%   x           Nxk matrix of independent variables
%
%   prior       structure with the parameters of the prior distribution:
%       beta   ~ N(prior.mu, prior.A^(-1) * sige^2)
%       sige^2 ~ Inverse Gamma (prior.a, prior.b)
%
%   options     structure with options for the MCMC estimator:
%       G_burnin    # burn-in draws 
%       G_samples   # draws after burn-in
%       seed        random number seed
%
% Returns the structure output, which contains a the sequence of draws of 
% beta and sige^2 from the MCMC chain.
%
% NB: the output draws include the burn-in draws.


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Initialize random number generator
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
s = RandStream('mcg16807', 'Seed', options.seed);
RandStream.setDefaultStream(s);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% initialize the variables that will hold the draws from the MCMC
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
beta_save  = zeros(options.G_burnin + options.G_samples, size(x,2));
sige2_save = zeros(options.G_burnin + options.G_samples, 1);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% run the MCMC loop
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for g = 1:(options.G_burnin + options.G_samples)
       
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % draw from distribution f(beta, sige^2 | y, x)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % This is a standard Bayesian regression (see Bayesregr.m for details)
    post = Bayesregr (y, x, prior);
    % draw error variance from the posterior Inverse Gamma
    sige2_draw = 1/gaminv(rand, post.a, 1/post.b);          
    % draw beta from the posterior multivariate Normal
    beta_draw  = mvnrnd(post.mu, sige2_draw * (post.A \ eye(size(x,2))) ); 
    
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % save draws
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    beta_save(g,:)  = beta_draw;
    sige2_save(g,1) = sige2_draw;

end


% Fill the output structure
output.beta  = beta_save;   
output.sige2 = sige2_save;
