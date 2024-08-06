function cdf=invgamcdf(X,A,B)
cdf=gammainc(B./X,A,'upper');

end