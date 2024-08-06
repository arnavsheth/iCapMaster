function output = MCMC_Heckman_randomeffects (y, x, z, firmid, prior, options)
% MCMC random effects panel regression model with selection.
%
% Copyright (C) 2011 by Arthur Korteweg
%
% This is Algorithm 6 in the book chapter
% "Markov Chain Monte Carlo Methods in Corporate Finance", 
%   In P. Damien, P. Dellaportas, N. Polson, and D. Stephens (Eds.), 
%   MCMC and  Hierarchical Models. Oxford University Press.
%
%
% The model is:
%   y(it) = alpha(i) + x(it)'*beta + epsilon(it)    epsilon ~ N(0, sige^2)     observation eqn
%   P(D(it)=1|z) = Phi(theta(i) + z(it)'*gamma)     Phi(.) is N(0,1) cdf       selection eqn
%
%   where D = 1 if y is observed, and D = 0 otherwise.
%
%   alpha and theta are independent random effects.
%   alpha(i) ~ N(mu, tau^2)
%   theta(i) ~ N(kappa, omega^2)
%
%   NB: time fixed effects can be incorporated by including them in x and z.
%
%   The error terms epsilon and eta are correlated with coefficient rho.
%
%   y           Nx1 vector containing the dependent variable
%               missing y-values are indicated with NaN.            
%   x           Nxk matrix of independent variables in observation equation
%   z           Nxk matrix of independent variables in selection equation
%   firmid      NTx1 vector of firm id's
%
%   The estimation uses the fact that epsilson = sige * (rho*eta + sqrt(1-rho^2)*xi)
%   with (eta, xi) iid N(0,1) random variables, to recast the model:
%           
%   y(it) = alpha(i) + x(it)'*beta + eta(it)*delta + sigxi*xi(it)    observation eqn
%   w(it) = theta(i) + z(it)'*gamma + sigeta*eta(it)                 selection eqn
%
%   delta  = sige * rho             i.e. the covariance between epsilson and eta
%   sigxi  = sige * sqrt(1-rho^2)   i.e. the cond. stdev of epsilon|eta
%   sigeta = sqrt(1 - omega^2)      i.e. the variance of theta+eta = 1
%
%   D(it) = I(w(it)>0)      I is an indicator variable (=1 if w>0 and 0 otherwise)
%
%   prior       structure with the parameters of the prior distribution:
%       beta    ~ N(prior.mu_beta, prior.A_beta^(-1) * sigxi^2)
%       delta   ~ N(prior.mu_delta, prior.A_delta^(-1) * sigxi^2)
%       gamma   ~ N(prior.mu_gamma, prior.A_gamma^(-1))
%       sige^2  ~ Inverse Gamma (prior.a, prior.b)
%
%   and "hyperpriors" for the random effect alpha
%       mu      ~ N(prior.mu_mu, prior.A_mu^(-1) * tau^2)
%       tau^2   ~ Inverse Gamma (prior.a_tau, prior.b_tau)
%       kappa   ~ N(prior.mu_kappa, prior.A_kappa^(-1) * tau^2)
%       omega^2 ~ Inverse Gamma (prior.a_omega, prior.b_omega)
%
%   options     structure with options for the MCMC estimator:
%       G_burnin    # burn-in draws 
%       G_samples   # draws after burn-in
%       seed        random number seed
%
% Returns the structure output, which contains the sequence of draws of 
% beta, gamma, sige^2, mu, tau^2, kappa, omega^2, rho and the two random 
% effects (alpha_i and theta_i) from the MCMC chain.
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
data = sortrows([firmid y x z], 1);
firmid = data(:,1);
y = data(:,2);
x = data(:,3:3+size(x,2)-1);
z = data(:,3+size(x,2):end);
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
alpha_save = zeros(options.G_burnin + options.G_samples, N);
beta_save  = zeros(options.G_burnin + options.G_samples, size(x,2));
theta_save = zeros(options.G_burnin + options.G_samples, N);
gamma_save = zeros(options.G_burnin + options.G_samples, size(z,2));
sige2_save = zeros(options.G_burnin + options.G_samples, 1);
rho_save   = zeros(options.G_burnin + options.G_samples, 1);
mu_save    = zeros(options.G_burnin + options.G_samples, 1);
tau2_save  = zeros(options.G_burnin + options.G_samples, 1);
kappa_save  = zeros(options.G_burnin + options.G_samples, 1);
omega2_save = zeros(options.G_burnin + options.G_samples, 1);


% set starting draws of the model parameters
y_draw     = y; y_draw(isnan(y)) = 0;
w_draw     = zeros(size(y));
alpha_draw = zeros(1,N);
beta_draw  = zeros(1,size(x,2));
theta_draw = zeros(1,N);
gamma_draw = zeros(1,size(z,2));
delta_draw = 0;             % start at rho = 0
sigxi2_draw = nanvar(y);    % consistent w/ rho=0 and beta=0
mu_draw     = prior.mu_mu;
tau2_draw   = nanvar(y)/2;    % so var(epsilon) = tau^2 + sige^2
kappa_draw  = prior.mu_kappa;
omega2_draw = 0.1;    


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% run the MCMC loop
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for g = 1:(options.G_burnin + options.G_samples)
    
    if mod(g,100) == 0, disp(['Drawing MCMC cycle ' num2str(g)]); end        % display progress
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % draw from distribution f(w, y* | alpha, beta, theta, gamma, delta, 
    %       sige^2, sigxi^2, rho, mu, tau^2, kappa, omega^2, y, z, x)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    for i = 1:length(y)        
        if isnan(y(i))
            % draw w for observations where y not observed: upper-truncated Normal
            mu        = firmdums(i,:)*theta_draw' + z(i,:)*gamma_draw';
            w_draw(i) = randn_uppertrunc(0, mu, sqrt(1-omega2_draw));
            % draw y*, the latent y variable when w < 0
            y_draw(i) = firmdums(i,:)*alpha_draw' + x(i,:)*beta_draw' + (w_draw(i) - mu)*delta_draw + sqrt(sigxi2_draw)*randn;
        else
            % draw w for observations where y observed: lower-truncated Normal    
            % Note that w|y ~ N( theta + z'*gamma + rho*sigeta*((y-alpha-x'*beta)/sige), sqrt(sigeta^2*(1-rho^2)) )
            %   since rho = delta/sige we can write sqrt(sigeta^2*(1-rho^2)) = sigeta*sigxi/sige
            sige2_draw = sigxi2_draw + delta_draw^2;
            mu        = firmdums(i,:)*theta_draw' + z(i,:)*gamma_draw' + (delta_draw*sqrt(1-omega2_draw)/sige2_draw) * (y(i) - firmdums(i,:)*alpha_draw' - x(i,:)*beta_draw');
            w_draw(i) = -randn_uppertrunc(0, -mu, sqrt((1-omega2_draw)*sigxi2_draw/sige2_draw) );           
        end
    end
         

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % draw from distribution f(beta, gamma | alpha, theta, delta, sigxi^2, 
    %       sige^2, rho, mu, tau^2, kappa, omega^2, y, z, x)
    %       This is a standard Bayesian SUR regression (e.g. Zellner, 1971)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    sigeta2_draw = 1-omega2_draw;
    Sigma       = [sigxi2_draw + delta_draw^2 sqrt(sigeta2_draw)*delta_draw; sqrt(sigeta2_draw)*delta_draw sigeta2_draw];
    Sinv        = inv(Sigma);
    prior.mu_B  = [prior.mu_beta; prior.mu_gamma];
    prior.A_B   = [prior.A_beta zeros(length(prior.mu_beta),length(prior.mu_gamma)); zeros(length(prior.mu_gamma),length(prior.mu_beta)) prior.A_gamma];  
    post.A_B    = prior.A_B + [Sinv(1,1)*x'*x Sinv(1,2)*x'*z; Sinv(2,1)*z'*x Sinv(2,2)*z'*z];   %= prior.A_B + X'*inv(kron(Sigma, eye(length(y))))*X;    
    post.mu_B   = post.A_B \ (prior.A_B * prior.mu_B + [Sinv(1,1)*x'*[y_draw - firmdums*alpha_draw'] + Sinv(1,2)*x'*[w_draw - firmdums*theta_draw']; Sinv(2,1)*z'*[y_draw - firmdums*alpha_draw'] + Sinv(2,2)*z'*[w_draw - firmdums*theta_draw']]);
    B_draw      = mvnrnd( post.mu_B, post.A_B \ eye(size(post.A_B,2)) );
    beta_draw   = B_draw(1:size(x,2));
    gamma_draw  = B_draw(size(x,2)+1:end);
    
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % draw from distribution f(alpha | beta, theta, delta, sigxi^2, sige^2, 
    %       rho, mu, tau^2, kappa, omega^2, y, z, x)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
    % This is a standard Bayesian regression on a constant with prior  
    % variance equal to tau^2, and known error variance sige^2.
    sige2_draw = sigxi2_draw + delta_draw^2;
    prior_alpha.A = eye(N) * sige2_draw/tau2_draw;  % s.t. prior variance is tau^2
    prior_alpha.mu = ones(N,1) * mu_draw;%zeros(N,1);
    post.A = prior_alpha.A + firmdums_firmdums;
    post.mu = post.A \ (prior_alpha.A * prior_alpha.mu + firmdums'*(y_draw - x*beta_draw') );    
    % draw alpha from the posterior multivariate Normal
    alpha_draw  = mvnrnd(post.mu, sige2_draw * (post.A \ eye(N)) ); 
    
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % draw from distribution f(theta | alpha, beta, delta, sigxi^2, sige^2, 
    %       rho, mu, tau^2, kappa, omega^2, y, z, x)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
    % This is a standard Bayesian regression on a constant with prior
    % variance equal to omega^2 and known error variance sigeta^2.
    sigeta2_draw = 1-omega2_draw;
    prior_theta.A = eye(N) * sigeta2_draw/omega2_draw;  % s.t. prior variance is omega^2
    prior_theta.mu = ones(N,1)*kappa_draw;%zeros(N,1);
    post.A = prior_theta.A + firmdums_firmdums;
    post.mu = post.A \ (prior_theta.A * prior_theta.mu + firmdums'*(w_draw - z*gamma_draw') );    
    % draw theta from the posterior multivariate Normal
    theta_draw  = mvnrnd(post.mu, sigeta2_draw * (post.A \ eye(N)) );     
    
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % draw from distribution f(mu, tau^2 | alpha, beta, theta, delta, 
    %       sigxi^2, sige^2, rho, kappa, omega^2, y, z, x)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
    % This is a standard Bayesian regression on a constant
    prior_mutau.A  = prior.A_mu;
    prior_mutau.mu = prior.mu_mu;
    prior_mutau.a  = prior.a_tau;
    prior_mutau.b  = prior.b_tau;
    post           = Bayesregr(alpha_draw', ones(N,1), prior_mutau);
    % draw tau^2 from the posterior Inverse Gamma
    tau2_draw      = 1/gaminv(rand, post.a, 1/post.b);          
    % draw mu from the posterior multivariate Normal
    mu_draw        = post.mu + sqrt(tau2_draw / post.A) * randn;
    
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % draw from distribution f(kappa, omega^2 | alpha, beta, theta, delta, 
    %       sigxi^2, sige^2, rho, mu, tau^2, y, z, x)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
    % This is a standard Bayesian regression on a constant
    prior_kappaomega.A  = prior.A_kappa;
    prior_kappaomega.mu = prior.mu_kappa;
    prior_kappaomega.a  = prior.a_omega;
    prior_kappaomega.b  = prior.b_omega;
    post                = Bayesregr(theta_draw', ones(N,1), prior_kappaomega);
    % draw tau^2 from the posterior Inverse Gamma
    omega2_draw         = 1/gaminv(rand, post.a, 1/post.b);          
    % draw mu from the posterior multivariate Normal
    kappa_draw          = post.mu + sqrt(omega2_draw / post.A) * randn;
        
            
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % draw from distribution f(delta, sigxi^2 | alpha, beta, gamma, mu, 
    %       tau^2, kappa, omega^2, sige^2, rho, y, z, x)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%       
    % This is a standard Bayesian regression (see Bayesregr.m fror details)
    eta = w_draw - firmdums*theta_draw' - z*gamma_draw';    
    prior_delta.A  = prior.A_delta;
    prior_delta.mu = prior.mu_delta;
    prior_delta.a  = prior.a;
    prior_delta.b  = prior.b;
    post           = Bayesregr (y_draw - firmdums*alpha_draw' - x*beta_draw', eta, prior_delta);
    % draw error variance from the posterior Inverse Gamma
    sigxi2_draw    = 1/gaminv(rand, post.a, 1/post.b);          
    % draw beta from the posterior multivariate Normal
    delta_draw     = post.mu + sqrt(sigxi2_draw / post.A) * randn;
    
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % save draws
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    alpha_save(g,:) = alpha_draw;
    beta_save(g,:)  = beta_draw;
    theta_save(g,:) = theta_draw;
    gamma_save(g,:) = gamma_draw;
    % translate (delta, sigxi) => (rho, sige)
    sige2_draw = sigxi2_draw + delta_draw^2;
    sige2_save(g,1) = sige2_draw;
    rho = delta_draw/ sqrt(sige2_draw);
    rho_save(g,1)   = delta_draw/ sqrt(sige2_draw);      
    mu_save(g,1)    = mu_draw;
    tau2_save(g,1)  = tau2_draw;
    kappa_save(g,1) = kappa_draw;
    omega2_save(g,1) = omega2_draw;

end

% Fill the output structure
output.alpha  = alpha_save;
output.beta   = beta_save; 
output.theta  = theta_save;
output.gamma  = gamma_save;   
output.sige2  = sige2_save;
output.rho    = rho_save;
output.mu     = mu_save;
output.tau2   = tau2_save;
output.kappa  = kappa_save;
output.omega2 = omega2_save;
