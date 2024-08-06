% Log Likelihood for all funds
%
% Moment equation is
% PV(div)/PV(invest) ~ lognormal(mu,sigma) with mean 1. 
%
% Outputs
% LL: log likelihood
% log of PVratios (varargout)

function [LL,PME,sig,mpme] = LL_allfunds_Norm(div,invest,g_ret,max_sig,mpme)

if nargin == 4
    mpme = 0; % lump-sum for PME
end
% Matrices of factors and factor loadings
% Each of size T x Num_Funds

Num_Funds = size(div,2);                                             % cols = number of funds
T = size(div,1) - 1;                                                 % CF_mat has one extra col than the time vector
r_mat = repmat(g_ret,1,Num_Funds);                                   % T x Num_Funds

% Compute cumulative returns from time zero
discount_mat = [ones(1,Num_Funds); 1./(1+r_mat)];                    % (T+1) x Num_Funds
discount_mat = cumprod(discount_mat);

% PVs 
PVdiv = sum(discount_mat.*div);                                      % 1 x Num_Funds
PVinvest = sum(discount_mat.*invest);

x = PVdiv./PVinvest;                                                 % Num_funds x 1.  Assume this is log normally distributed
x = x';

% Demean
% Assume the distribution is centered at 1.
% The mean of the log normal is exp(.5*sig^2). Hence, mu = -.5*sig^2;

sig=min(std(log(x)),max_sig+randn*0.01);
mu =  .5 * sig^2 ;

% Construct log likelihood of log normal
LL = log(1./(x*sig)) - .5/sig^2*(log(x)-mu).^2; 
LL = sum(LL);

% Output log of PV ratios
PME = x;   % sample PME

% Sample lump-sum we allowed for log(PME)
mpme = mean(log(PME));    

% mpme = log(mean(PME))-0.5*sig^2;              % mean(exp(log(PME)) = exp(mean(log(PME))+0.5sig^2 ) 
                                              % mean(log(PME)) = log(mean(PME)) - 0.5 sig^2
                                              % the mean of log(PME) need to be demeaned by (log(mean(PME))-0.5 sig^2)
                                              