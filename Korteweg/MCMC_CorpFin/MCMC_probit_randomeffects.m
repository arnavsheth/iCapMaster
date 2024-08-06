function output = MCMC_probit_randomeffects (y, x, firmid, prior, options)
% MCMC random effects panel probit regression of y on x.
%
% Copyright (C) 2011 by Arthur Korteweg
%
% This is Algorithm 5 in the book chapter
% "Markov Chain Monte Carlo Methods in Corporate Finance", 
%   In P. Damien, P. Dellaportas, N. Polson, and D. Stephens (Eds.), 
%   MCMC and  Hierarchical Models. Oxford University Press.
%
%
% The model is:
%   P(y(it)=1|x) = Phi(alpha(i) + x(it)'*beta)     where Phi(.) is the standard Normal cdf
%
%   y           NTx1 vector containing the dummy dependent variable (0 or 1)       
%   x           NTxk matrix of independent variables
%   firmid      NTx1 vector of firm id's
%
%   The estimation uses the recasted probit model:
%           
%   y(it) = I(w(it)>0)                              observation equation 
%   w(it) = x(it)'*beta + eta(i)   eta(i) ~ N(0,1)    selection equation
%
%   I(.) is the indicator function
%
%   The error term eta contains the fixed effect:
%       eta(i) = alpha(i) + u(it)       
%
%       where alpha(i) ~ N(0, tau^2) and u(it) are iid N(0, 1 - tau^2) 
%
%   prior       structure with the parameters of the prior distribution:
%       beta    ~ N(prior.mu_beta, prior.A_beta^(-1) * 1)
%
%   and "hyperpriors" for the random effect alpha
%       tau^2   ~ Inverse Gamma (prior.a_tau, prior.b_tau)
%
%   options     structure with options for the MCMC estimator:
%       G_burnin    # burn-in draws 
%       G_samples   # draws after burn-in
%       seed        random number seed
%
% Returns the structure output, which contains the sequence of draws of 
% beta, tau^2 and the alpha_i from the MCMC chain.
%
% NB: the output draws include the burn-in draws.


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Initialize random number generator
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
s = RandStream('mcg16807', 'Seed', options.seed);
RandStream.setDefaultStream(s);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Ensure the data is sorted by firm id's
% Construct a matrix of dummy variables for the (random) intercepts
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
data = sortrows([firmid y x], 1);
firmid = data(:,1);
y = data(:,2);
x = data(:,3:end);
% count number of firms 
N = 0; for i = 1:max(firmid), if sum(firmid==i)>0, N = N + 1; end; end  
firmdums = zeros(length(y), N);
index = 1;
for i = 1:max(firmid)
    if sum(firmid==i)>0     % make sure this firm id exists
        firmdums(firmid==i, index) = 1;
        index = index + 1;
    end
end
firmdums_firmdums = firmdums'*firmdums;     % speeds up calculations below


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% initialize the variables that will hold the draws from the MCMC
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
beta_save  = zeros(options.G_burnin + options.G_samples, size(x,2));
tau2_save  = zeros(options.G_burnin + options.G_samples, 1);
alpha_save = zeros(options.G_burnin + options.G_samples, N);

% set starting draws of the model parameters
w_draw     = zeros(size(y));
beta_draw  = zeros(1,size(x,2));
tau2_draw  = 0.5;    
alpha_draw = zeros(1,N);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% run the MCMC loop
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for g = 1:(options.G_burnin + options.G_samples)
    
    if mod(g,100) == 0, disp(['Drawing MCMC cycle ' num2str(g)]); end        % display progress
          
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % draw from distribution f(w | alpha, beta, tau^2, x)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    for i = 1:length(y)
        mu = firmdums(i,:)*alpha_draw' + x(i,:)*beta_draw';
        if y(i) == 0 
            % draw w for observations where y = 0: upper-truncated Normal
            w_draw(i) = randn_uppertrunc(0, mu, 1);
        else
            % draw w for observations where y = 1: lower-truncated Normal            
            w_draw(i) = -randn_uppertrunc(0, -mu, 1);            
        end
    end

    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % draw from distribution f(beta | alpha, tau^2, w, y, x)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
    % This is a standard Bayesian regression with known error variance
    % equal to 1-tau^2.
    post.A = prior.A + x' * x;
    post.mu = post.A \ (prior.A * prior.mu + x'*(w_draw - firmdums*alpha_draw') );    
    % draw beta from the posterior multivariate Normal
    beta_draw  = mvnrnd(post.mu, (1-tau2_draw) * (post.A \ eye(size(x,2))) );

    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % draw from distribution f(alpha | beta, tau^2, w, y, x)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
    % This is a standard Bayesian regression on a constant with prior 
    % variance equal to tau^2 and known error variance 1-tau^2.
    prior_alpha.A = eye(N) * (1 - tau2_draw)/tau2_draw;  % s.t. prior variance is tau^2
    prior_alpha.mu = zeros(N,1);
    post.A = prior_alpha.A + firmdums_firmdums;
    post.mu = post.A \ (prior_alpha.A * prior_alpha.mu + firmdums'*(w_draw - x*beta_draw') );    
    % draw alpha from the posterior multivariate Normal
    alpha_draw  = mvnrnd(post.mu, (1-tau2_draw) * (post.A \ eye(N)) ); 
    
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % draw from distribution f(tau^2 | alpha, beta, w, y, x)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
    % This is a standard Bayesian regression with a known mean (= zero)
    e  = alpha_draw';
    S = e' * e;
    post.a = prior.a + N;
    post.b = prior.b + S;    
    tau2_draw = 1/gaminv(rand, post.a, 1/post.b);
               
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % save draws
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    alpha_save(g,:) = alpha_draw;
    beta_save(g,:)  = beta_draw;
    tau2_save(g,:)  = tau2_draw;

end

% Fill the output structure
output.alpha = alpha_save;   
output.beta  = beta_save;   
output.tau2  = tau2_save;
