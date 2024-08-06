function output = MCMC_Heckman (y, x, z, prior, options)
% MCMC Heckman selection model.
%
% Copyright (C) 2011 by Arthur Korteweg
%
% This is Algorithm 4 in the book chapter
% "Markov Chain Monte Carlo Methods in Corporate Finance", 
%   In P. Damien, P. Dellaportas, N. Polson, and D. Stephens (Eds.), 
%   MCMC and  Hierarchical Models. Oxford University Press.
%
%
% The model is:
%   y = x'*beta + epsilon        epsilon ~ N(0, sige^2)     observation eqn
%   P(D=1|z) = Phi(z'*gamma)     Phi(.) is N(0,1) cdf       selection eqn
%
%   where D = 1 if y is observed, and D = 0 otherwise
%
%   The error terms epsilon and eta are correlated with coefficient rho
%
%   y           Nx1 vector containing the dependent variable
%               missing y-values are indicated with NaN.            
%   x           Nxk matrix of independent variables in observation equation
%   z           Nxk matrix of independent variables in selection equation
%
%   The estimation uses the fact that epsilson = sige * (rho*eta + sqrt(1-rho^2)*xi)
%   with (eta, xi) independent N(0,1) random variables, to recast the model:
%           
%   y = x'*beta + eta*delta + sigxi*xi        observation eqn
%   w = z'*gamma + eta                        selection eqn
%
%   delta  = sige * rho     i.e. the covariance between epsilson and eta
%   sigxi = sige * sqrt(1-rho^2)    i.e. the cond. stdev of epsilon|eta
%
%   D = I(w>0)      I is an indicator variable (=1 if w>0 and 0 otherwise)
%
%   prior       structure with the parameters of the prior distribution:
%       beta    ~ N(prior.mu_beta, prior.A_beta^(-1) * sigxi^2)
%       delta   ~ N(prior.mu_delta, prior.A_delta^(-1) * sigxi^2)
%       gamma   ~ N(prior.mu_gamma, prior.A_gamma^(-1))
%       sige^2  ~ Inverse Gamma (prior.a, prior.b)
%
%   options     structure with options for the MCMC estimator:
%       G_burnin    # burn-in draws 
%       G_samples   # draws after burn-in
%       seed        random number seed
%
% Returns the structure output, which contains the sequence of draws of 
% beta, gamma, sige^2 and rho from the MCMC chain.
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
gamma_save = zeros(options.G_burnin + options.G_samples, size(z,2));
sige2_save = zeros(options.G_burnin + options.G_samples, 1);
rho_save   = zeros(options.G_burnin + options.G_samples, 1);
% the next line is optional, if you want to look at the selection variable:
% w_save = zeros(options.G_burnin + options.G_samples, length(y));


% set starting draws of the model parameters
y_draw     = y; y_draw(isnan(y)) = 0;
w_draw     = zeros(size(y));
beta_draw  = zeros(1,size(x,2));
gamma_draw = zeros(1,size(x,2));
delta_draw = 0;             % start at rho = 0
sigxi2_draw = nanvar(y);    % consistent w/ rho=0 and beta=0


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% run the MCMC loop
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for g = 1:(options.G_burnin + options.G_samples)
    
    if mod(g,1000) == 0, disp(['Drawing MCMC cycle ' num2str(g)]); end        % display progress
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % draw from distribution f(w, y* | beta, gamma, delta, sige^2, sigxi^2, rho, y, z, x)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    for i = 1:length(y)        
        if isnan(y(i))
            % draw w for observations where y not observed: upper-truncated Normal
            mu        = z(i,:)*gamma_draw';
            w_draw(i) = randn_uppertrunc(0, mu, 1);
            % draw y*, the latent y variable when w < 0
            y_draw(i) = x(i,:)*beta_draw' + (w_draw(i) - mu)*delta_draw + sqrt(sigxi2_draw)*randn;
        else
            % draw w for observations where y observed: lower-truncated Normal    
            % Note that w|y ~ N( z'*gamma + rho*((y-x'*beta)/sige), sqrt(1-rho^2) )
            %   since rho = delta/sige we can write sqrt(1-rho^2) = sigxi/sige
            sige2_draw = sigxi2_draw + delta_draw^2;
            mu        = z(i,:)*gamma_draw' + (delta_draw/sige2_draw) * (y(i) - x(i,:)*beta_draw');
            w_draw(i) = -randn_uppertrunc(0, -mu, sqrt(sigxi2_draw/sige2_draw));           
        end
    end         


    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % draw from distribution f(beta, gamma | delta, sigxi^2, y, w, z, x)
    %       This is a standard Bayesian SUR regression (e.g. Zellner, 1971)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    Sigma       = [sigxi2_draw + delta_draw^2 delta_draw; delta_draw 1];  
    Sinv        = inv(Sigma);
    prior.mu_B  = [prior.mu_beta; prior.mu_gamma];
    prior.A_B   = [prior.A_beta zeros(length(prior.mu_beta),length(prior.mu_gamma)); zeros(length(prior.mu_gamma),length(prior.mu_beta)) prior.A_gamma];  
    post.A_B    = prior.A_B + [Sinv(1,1)*x'*x Sinv(1,2)*x'*z; Sinv(2,1)*z'*x Sinv(2,2)*z'*z];   %= prior.A_B + X'*inv(kron(Sigma, eye(length(y))))*X;    
    post.mu_B   = post.A_B \ (prior.A_B * prior.mu_B + [Sinv(1,1)*x'*y_draw + Sinv(1,2)*x'*w_draw; Sinv(2,1)*z'*y_draw + Sinv(2,2)*z'*w_draw]);
    B_draw      = mvnrnd( post.mu_B, post.A_B \ eye(size(post.A_B,2)) );  
    beta_draw   = B_draw(1:size(x,2));
    gamma_draw  = B_draw(size(x,2)+1:end);   
    
        
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % draw from distribution f(delta, sigxi^2 | beta, gamma, y, w, z, x)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%       
    % This is a standard Bayesian regression (see Bayesregr.m fror details)
    eta = w_draw - z*gamma_draw';    
    prior_delta.A  = prior.A_delta;
    prior_delta.mu = prior.mu_delta;
    prior_delta.a  = prior.a;
    prior_delta.b  = prior.b;
    post           = Bayesregr (y_draw - x*beta_draw', eta, prior_delta);
    % draw error variance from the posterior Inverse Gamma
    sigxi2_draw    = 1/gaminv(rand, post.a, 1/post.b);          
    % draw beta from the posterior multivariate Normal
    delta_draw     = post.mu + sqrt(sigxi2_draw / post.A) * randn;
    
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % save draws
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    beta_save(g,:)  = beta_draw;
    gamma_save(g,:) = gamma_draw;
    % translate (delta, sigxi) => (rho, sige)
    sige2_draw = sigxi2_draw + delta_draw^2;
    sige2_save(g,1) = sige2_draw;
    rho = delta_draw/ sqrt(sige2_draw);
    rho_save(g,1)   = delta_draw/ sqrt(sige2_draw);      
    % the next line is optional, if you want to look at the selection variable:    
    % w_save(g,:) = w_draw';
    
end

% Fill the output structure
output.beta  = beta_save;   
output.gamma = gamma_save;   
output.sige2 = sige2_save;
output.rho   = rho_save;
% the next line is optional, if you want to look at the selection variable:
% output.w=w_save;
