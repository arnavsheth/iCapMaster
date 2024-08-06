function output = MCMC_probit (y, x, prior, options)
% MCMC probit regression of y on x.
%
% Copyright (C) 2011 by Arthur Korteweg
%
% This is Algorithm 3 in the book chapter
% "Markov Chain Monte Carlo Methods in Corporate Finance", 
%   In P. Damien, P. Dellaportas, N. Polson, and D. Stephens (Eds.), 
%   MCMC and  Hierarchical Models. Oxford University Press.
%
%
% The model is:
%   P(y=1|x) = Phi(x'*beta)     where Phi(.) is the standard Normal cdf
%
%   y           Nx1 vector containing the dummy dependent variable (0 or 1)
%   x           Nxk matrix of independent variables
%
%   The estimation uses the recasted probit model:
%           
%   y = I(w>0)                                  observation equation 
%   w = x'*beta + eta       eta ~ iid N(0,1)    selection equation
%
%   I(.) is the indicator function
%
%   prior       structure with the parameters of the prior distribution:
%       beta ~ N(prior.mu, prior.A^(-1))
%
%   options     structure with options for the MCMC estimator:
%       G_burnin    # burn-in draws 
%       G_samples   # draws after burn-in
%       seed        random number seed
%
% Returns the structure output, which contains the sequence of draws of 
% beta from the MCMC chain.
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

w_draw     = zeros(size(y));
beta_draw  = zeros(1,size(x,2));


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% run the MCMC loop
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for g = 1:(options.G_burnin + options.G_samples)
    
    if mod(g,1000) == 0, disp(['Drawing MCMC cycle ' num2str(g)]); end        % display progress
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % draw from distribution f(w | beta, x)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    for i = 1:length(y)
        mu = x(i,:)*beta_draw';
        if y(i) == 0 
            % draw w for observations where y = 0: upper-truncated Normal
            w_draw(i) = randn_uppertrunc(0, mu, 1);
        else
            % draw w for observations where y = 1: lower-truncated Normal            
            w_draw(i) = -randn_uppertrunc(0, -mu, 1);            
        end
    end

    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % draw from distribution f(beta | w, x)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % This is a standard Bayesian regression with known error variance
    % equal to 1.
    post.A = prior.A + x' * x;
    post.mu = post.A \ (prior.A * prior.mu + x'*w_draw);    
    % draw beta from the posterior multivariate Normal
    beta_draw  = mvnrnd(post.mu, (post.A \ eye(size(x,2))) ); 
    
    
    % save draws
    beta_save(g,:)  = beta_draw;

end

% Fill the output structure
output.beta = beta_save;   

