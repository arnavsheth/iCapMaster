%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Simdata.m
% 
% Copyright (C) 2009 by Arthur Korteweg and Morten Sorensen
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% This program simulates the dynamic selection model from the
% paper "Risk and Return Characteristics of Venture Capital 
% Investments in Entrepreneurial Companies".
%    v_t = v_(t-1) + rf + delta + beta*rmrf + e_t			(valuation eqn)
%    W_t = Z_t * gamma_0 + (v_t - vF_t) * gamma_v + eta_t	(selection eqn)
% 
% where e(t) ~ N(0, sigma^2) i.i.d.
% 	  eta(t) ~ N(0,1) i.i.d.
% and e(t) and eta(t) are independent
% 
% vF_t is the log-valuation at the previous financing round.
% 
% Z_t = [1 tau tau^2] where tau is the number of periods since the previous 
% observed valuation (i.e. financing round).
% 
% rf is assumed constant.
% 
% Simulated data is saved in the m-file "Simdata.mat" for estimating the 
% model in Matlab (using "MCMC.m"), and in 3 files for use in estimating the
% model in C++ (using "MCMC.cpp"):
% 	Simdata_info.dat		T (# periods), N (# firms), risk-free rate rf
% 	Simdata_logVobs.dat		log-valuations (TxN matrix), unobserved values are -999
% 	Simdata_rmrf.dat		rmrf (vector of length T-1)
 
clear all
close all

T = 120;                % # time periods in sample
N = 10;                 % # firms to simulate

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% CAVEAT on changing "true" model parameters
%
% It is possible to end up with an extremely poorly populated dataset by
% changing the true delta or the gamma loading on valuations (the second
% entry in true_gamma). For example, if you change true_delta to -0.1 you
% end up with only a couple of observations, since most firm values will 
% quickly drift down to zero, never to be observed again.
% When the problem is not identified (or nearly unidentified), the trace 
% plots in MCMC.m will seem to wander around aimlessly. 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

true_delta = 0;         % true market model delta
true_beta = 1;          % true market model beta
true_sige = 0.1;        % per-period idiosyncratic volatility

true_gamma = [-1; 10; 0.1; 0];  % true gamma
true_logV = ones(T,N);          % TxN matrix of log-valuations, start at 1
X_select = zeros(T,4,N);        % selection variables (4 per firm-period)
logVobs = zeros(T,N);           % log-valuations that are actually observed
logVF   = zeros(T,N);           % log-valuations at previous refinancing

sig_rmrf = 0.02;            % per-period volatility of excess market return
mu_rmrf = -sig_rmrf^2/2;    % per-period expected excess market return
rf = 0;                     % per-period risk-free rate (constant)


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Simulate the model
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Simulate the market return and value
rmrf = mu_rmrf + randn(T-1,1)*sig_rmrf;        % draw market return

% Simulate returns and values for the N firms
rirf = true_delta + kron(true_beta*rmrf, ones(1,N)) + randn(T-1,N)*true_sige; % draw security return

% Calculate true log-valuations (all starting at 1 at time 1)
true_logV(2:end,:) = 1 + cumsum(rf + rirf);

% Use the selection equation to determine which valuations are observed
for i = 1:N    
    tau = 0;                            % # periods since last refinancing
    logVF(1,i) = true_logV(1,i);        % log-valuation at previous observed round
    for t = 1:T
        true_W = X_select(t,:,i) * true_gamma + randn;  
        if t==1 || (true_W > 0)         % is V observed? NB: force first period to be observed (wlog)
            logVobs(t,i) = true_logV(t,i);  % record log(V) as observed
            logVF(t+1,i) = true_logV(t,i);  % update valuation since last refinancing
            tau = 1;                        % reset time since last refinancing
        else
            logVobs(t,i) = -999;        % no observation this month
            logVF(t+1,i) = logVF(t,i);  % no change in last observed valuation
            tau = tau + 1;              % increase # periods since last observed valuation
        end
        if t < T                        % fill in selection variables for next period
            X_select(t+1,:,i) = [1 true_logV(t+1,i)-logVF(t+1,i) tau tau^2];    
        end
    end     % for t    
end     % for i


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Save the data for use in c++ 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% save info: T, N, rf
save('Simdata_info.dat', 'T', 'N', 'rf', '-ascii');

% save logVobs, the observed log-valuations
y = [];
for i = 1:size(logVobs,1);
    y = [y; logVobs(i,:)'];     % save in row-order 
end
save('Simdata_logVobs.dat', 'y', '-ascii', '-double')

% save rmrf
save('Simdata_rmrf.dat', 'rmrf', '-ascii', '-double')


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Save the data for use in Matlab 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

save Simdata N T rf rmrf logVobs true_delta true_beta true_gamma true_sige