function y = randn_lowertrunc(bound, mu, sigma)
% draws from a lower-truncated normal with mean mu and standard 
% deviation sigma, truncated from below at bound.

y = 2*mu - randn_uppertrunc(2*mu-bound, mu, sigma);