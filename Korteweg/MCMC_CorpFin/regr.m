function out = regr(y, x);
% Copyright (C) 2011 by Arthur Korteweg
%
% Standard OLS regression of x on y.
% Reports output in the structure 'out'. Elements of 'out' are:
% b         coefficient estimate
% sb        standard errors of coefficients
% t         t-stats
% stde      standard deviation of error term
% R2        R-squared
% R2a       adjusted R-squared
% F         model F-test statistic
% p         p-value associated with the F-test

[N,K]   = size(x);
b       = x \ y;
yfit    = x * b;
e       = y - yfit;
vare    = e'*e / (N-K);
stde    = sqrt(vare);
Xinv    = (x'*x) \ eye(K);
sb      = sqrt(diag(Xinv * vare));
t       = b ./ sb;
R2      = 1 - e'*e/((y-mean(y))'*(y-mean(y)));
R2a     = 1 - (1-R2)*(N-1)/(N-K);
F       = R2/(1-R2) * (N-K)/(K-1);%((y-mean(y))'*(y-mean(y)) - e'*e)/(K-1) / vare ; %var(y)/var(e) * (N-1)/(K-1) - (N-K)/(K-1);%
p       = 1-fcdf(F, K-1, N-K);

% fill in the structure 'out'
out.b       = b;        % coefficient estimates
out.sb      = sb;       % standard errors of coefficients
out.t       = t;        % t-stats
out.stde    = stde;     % std dev of the residuals
out.R2      = R2;       % R-squared
out.R2a     = R2a;      % adjusted R-squared
out.F       = F;        % F-value  
out.p       = p;        % p-value