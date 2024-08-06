%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% MCMCout_sum.m
% 
% Copyright (C) 2009 by Arthur Korteweg and Morten Sorensen
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% This program presents the output from the C++ estimation in MCMC.cpp in a
% convenient way. See Simdata.m for a detailed description of the model. 
%
% The program summarizes the posterior mean and standard deviation of
% the MCMC algorithm in table format.
%
% Second, the program makes histograms of the posterior distributions of 
% delta, beta, sigma, and the gamma's. True values that were used to 
% generate the data are marked by red lines.
%
% Third, plot trace plots to show how the parameter draws change over
% iterations of the algorithm. The true value that were used to generate
% the data are marked by red lines.


clear all
close all

G_burn = 1000;          % number of iterations to discard for burn-in

load simdata            % to plot the true values that were used in to simulate the data

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Load c++ output
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
G = load('MCMCout_info.dat','-ascii');      

y = load('MCMCout_b.dat', '-ascii');
NF = length(y)/G;       % number of factors in pricing model (valuation equation)
for g = 1:G
    b(g,:) = y((g-1)*NF+1:g*NF);        % GxNF matrix, saved in row order
end

y = load('MCMCout_gamma.dat', '-ascii');
NK = length(y)/G;       % number of factors in selection equation
for g = 1:G
    gamma(g,:) = y((g-1)*NK+1:g*NK);    % GxNK matrix, saved in row order
end

sige2 = load('MCMCout_sige2.dat', '-ascii');    % Gx1 vector


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Summarize output in table format
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp(' ')
disp('Summary of MCMC posterior');
disp('b:     mean    s.d.    true value');
disp([mean(b(G_burn+1:end,:))' std(b(G_burn+1:end,:))' [true_delta; true_beta]]);
disp('sigma: mean    s.d.    true value');
x = sqrt(sige2);
disp([mean(x(G_burn+1:end)) std(x(G_burn+1:end)) true_sige]);
disp('gamma: mean    s.d.    true value');
x = gamma; 
disp([mean(x(G_burn+1:end,:))' std(x(G_burn+1:end,:))' true_gamma]);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Plot histograms 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure;
title('Posterior Histograms')       % 
subplot(2,2,1); histplot(b(G_burn+1:end,1),25,true_delta); title('\delta');     
subplot(2,2,2); histplot(b(G_burn+1:end,2),25,true_beta); title('\beta');
x = sqrt(sige2);
subplot(2,2,3); histplot(x(G_burn+1:end),25,true_sige); title('\sigma');

figure;
title('Posterior Histograms')
x = gamma'; 
for i = 1:min(NK,8)
    subplot(ceil(min(NK,8)/2),2,i); 
    histplot(x(i,G_burn+1:end),25,true_gamma(i)); 
    title(['\gamma_' num2str(i)]);
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Make trace plots
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure;
title('Parameter trace plots')
subplot(2,2,1); plot(b(:,1)); line([1 G+1], [true_delta true_delta], 'Color', 'r', 'LineWidth', 2); title('\delta'); v = axis; axis([1 G+1 v(3:4)]); xlabel('Iteration')
subplot(2,2,2); plot(b(:,2)); line([1 G+1], [true_beta true_beta], 'Color', 'r', 'LineWidth', 2); title('\beta'); v = axis; axis([1 G+1 v(3:4)]); xlabel('Iteration')
subplot(2,2,3); plot(sqrt(sige2)); line([1 G+1], [true_sige true_sige], 'Color', 'r', 'LineWidth', 2); title('\sigma'); v = axis; axis([1 G+1 v(3:4)]); xlabel('Iteration')

figure;
title('Parameter trace plots');
x = gamma';
for i = 1:min(NK,8)
    subplot(ceil(min(NK,8)/2),2,i); 
    plot(x(i,:)); 
    v = axis;
    axis([1 G+1 v(3:4)])
    set(gca,'XTick',0:G/5:G)
    line([1 G+1], [true_gamma(i) true_gamma(i)], 'Color', 'r', 'LineWidth', 2);
    title(['\gamma_' num2str(i)]);
end
