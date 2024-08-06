function X=invgaminv(P,A,B)
X=B./gammaincinv(P,A,'upper');

end