function posterior = Bayesregr (y, x, prior)
% Copyright (C) 2011 by Arthur Korteweg
%
% Posterior distribution of beta of a Bayesian regression 
%   y = x'*b + epsilon       epsilon ~ N(0, sigma^2)
%
%   y   Nx1 vector containing the dependent variable
%   x   Nxk matrix of the independent variables
%
%   prior       structure with the parameters of the prior distribution:
%       b       ~ N(prior.mu, prior.A^(-1) * sigma^2)
%       sigma^2 ~ Inverse Gamma (prior.a, prior.b)
%
%   Returns the parameters of the posterior in a structure:
%       b|data       ~ N(posterior.mu, posterior.A^(-1) * sigma^2)
%       sigma^2|data ~ Inverse Gamma (posterior.a, posterior.b)
%

posterior.A = prior.A + x' * x;
posterior.mu = posterior.A \ (prior.A * prior.mu + x'*y);

posterior.a = prior.a + size(y,1);

e  = y - x*posterior.mu;
S = e' * e + (posterior.mu - prior.mu)' * prior.A * (posterior.mu - prior.mu);
posterior.b = prior.b + S;
