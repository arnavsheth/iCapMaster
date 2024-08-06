function y = randn_uppertrunc(bound, mu, sigma)
% Copyright (C) 2011 by Arthur Korteweg
%
% draws from a upper-truncated normal with mean mu and standard 
% deviation sigma, truncated from above at bound.

zbound = (bound - mu) / sigma;   % translate to standard normal
z = standardnormal_uppertrunc(zbound);
y = mu + z * sigma;         % scale back to Normal(mu, sigma)


function x = standardnormal_uppertrunc(u)
% draws from an upper-truncated standard normal using either symmetric
% rejection, standard rejection or Geweke (1991) exponential rejection.

if (u < 0) && (u > -1)
    x = 10;
    while (x > u)
        x = randn;
        if (x > -u), x = -x; end
    end
elseif (u >= 0)
    x = u+1;
    while (x > u), x = randn; end
else
    z = log(rand) / u;
    while (rand > exp(-z*z/2))
        z = log(rand) / u;
    end
    x = u - z;
end