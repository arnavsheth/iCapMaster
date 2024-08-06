#################################################################################################
#  This is an empirical application of lineal panel data methods to the effect of spending on
#  student test pass rates similar to Papke (2005).  
#  14.382 L8 MIT.  V. Chernozhukov and I. Fernandez-Val
#
# Data source: Christian Hansen (Chicago Booth), N = 550 schooling districts, T = 7 years (1992-1998)
# first year is lost to construct lagged variables
#
# Description of the data: the sample selection and variable contruction follow
#
# Leslie E. Papke, The effects of spending on test pass rates: evidence from Michigan, Journal of 
# Public Economics, Volume 89, Issues 5–6, June 2005, Pages 821-839
#
# The variables in the data set include:
#
# distid = "school district identifier"
# year = "year"
# math4 = "fraction of 4th grade students receiving a passing score on a standarized test"
# rexpp = "real expenditure in 1997 dollars per pupil in the district"
# l1rexpp = "lag of rexpp"
# enrol = "district level enrollment"
# lunch = "fraction of students in the district eligible for free lunch program"
#################################################################################################


# Updated on 04/09/2016


##################################################################################################



### set working directory
filepathIvan<-"/Users/Ivan/Dropbox/Shared/14.382"
filepathVictor<-"/Users/VC/Dropbox/TEACHING/14.382"
filepathMert<-"/Users/mertdemirer1/Dropbox (Personal)/14.382"
filepathVira<-"/Users/virasemenova/Dropbox/14.382"
setwd(filepathIvan)
###  read-in TestScores data

library(foreign);
library(xtable);
library(plm);
library(gmm);

data <- read.dta("Data/TestScores/TestScores.dta")
data <- pdata.frame(data, index = c("distid","year"));

attach(data);


########## Descriptive statistics

options(digits=2);
dstats <- cbind(sapply(data, mean), sapply(data, sd));
dimnames(dstats)[[2]] <- c("Mean", "SD");
xtable(dstats);

########## OLS estimation

form.ols <- math4 ~ log(rexpp) + log(l1rexpp) + log(enrol) + lunch + factor(year);

ols.fit   <- plm(form.ols, data, model = "pooling", index = c("distid","year"));
coefs.ols <- coef(ols.fit); 
se.ols    <- summary(ols.fit)$coef[ ,2];
HCV.coefs <- vcovHC(ols.fit, method = 'arellano', type = 'HC0', cluster = 'group');
cse.ols   <- sqrt(diag(HCV.coefs)); # Clustered std errors

# ########## Random Effects estimation
# 
# form.re <- math4 ~ log(rexpp) + log(l1rexpp) + log(enrol) + lunch + factor(year);
# 
# re.fit   <- plm(form.re, data, model = "random", index = c("distid","year"));
# coefs.re <- coef(re.fit); 
# se.re    <- summary(re.fit)$coef[ ,2];
# HCV.coefs <- vcovHC(re.fit, method = 'arellano', type = 'HC0', cluster = 'group');
# cse.re   <- sqrt(diag(HCV.coefs)); # Clustered std errors
# 


########## Fixed Effects estimation

form.fe <- math4 ~ log(rexpp) + log(l1rexpp) + log(enrol) + lunch -1 ;

fe.fit    <- plm(form.fe, data, model = "within", effect = "twoways", index = c("distid","year"));
coefs.fe  <- coef(fe.fit); 
se.fe     <- summary(fe.fit)$coef[ ,2];
HCV.coefs <- vcovHC(fe.fit, cluster = 'group');
cse.fe    <- sqrt(diag(HCV.coefs)); # Clustered std errors

########## First Differences estimation

form.fd <- math4 ~ log(rexpp) + log(l1rexpp) + log(enrol) + lunch + I(year==1995) + I(year==1996) + I(year==1997) + I(year==1998) ;

fd.fit    <- plm(form.fd, data, model = "fd", index = c("distid","year"));
coefs.fd  <- coef(fd.fit); 
se.fd     <- summary(fd.fit)$coef[ ,2];
HCV.coefs <- vcovHC(fd.fit, method = 'arellano', type = 'HC0', cluster = 'group');
cse.fd    <- sqrt(diag(HCV.coefs)); # Clustered std errors


########## Panel bootstrap std errors


set.seed(8);
N <- length(unique(distid));
T <- length(unique(year));
R <- 500;
coefs.ols.b <- matrix(0, ncol = length(coefs.ols), nrow = R);
#coefs.re.b  <- matrix(0, ncol = length(coefs.re), nrow = R);
coefs.fe.b  <- matrix(0, ncol = length(coefs.fe), nrow = R);
coefs.fd.b  <- matrix(0, ncol = length(coefs.fd), nrow = R);

for (r in 1:R) {;
                ids                <- kronecker(sample.int(N, N, replace = TRUE), rep(1,T));
                data.b             <- data[(ids-1)*T + rep(c(1:T),N), ];
                data.b$distid      <- kronecker(c(1:N), rep(1,T));
                data.b$year        <- rep(c(1992:1998),N);
                data.b             <- data.frame(data.b);
                data.b             <- pdata.frame(data.b, index = c("distid","year"));         # reset indexes of the panel        
                coefs.ols.b[r, ]   <- coef(plm(form.ols, data.b, model = "pooling", index = c("distid","year")));
#                coefs.re.b[r, ]   <- coef(plm(form.re, data.b, model = "random", index = c("distid","year")));
                coefs.fe.b[r, ]    <- coef(plm(form.fe, data.b, model = "within", effect = "twoways", index = c("distid","year")));
                coefs.fd.b[r, ]    <- coef(plm(form.fd, data.b, model = "fd", index = c("distid","year")));  
};

bse.ols <- apply(coefs.ols.b, 2, sd);
#bse.re  <- apply(coefs.re.b, 2, sd);
bse.fe  <- apply(coefs.fe.b, 2, sd);
bse.fd  <- apply(coefs.fd.b, 2, sd);


#############################################################################################;
#     GMM ESTIMATION;
#############################################################################################;

# Setting up data set;

# Dropping first year because l1rexpp has NA;

rdata <- data[year != 1992, ];

detach(data);


#### GMM 1: USING CONTEMPORANEOUS MOMENT CONDITIONS


# Functions to implement GMM estimators;

# First-step estimator;

fe.gmm <- function(Ddata, P, Tp)
{
  x <- Ddata[,-c(1:Tp)];
  y <- Ddata[,c(1:Tp)];  
  N <- nrow(x);
  sxx <- matrix(0, nrow = P, ncol = P);
  sxy <- matrix(0, nrow = P, ncol = 1);
  
  for (i in 1:nrow(x))
  {
    xi <- matrix(x[i, ], nrow = Tp);
    yi <- matrix(y[i, ], nrow = Tp);
    sxx <- sxx + t(xi) %*% xi;
    sxy <- sxy + t(xi) %*% yi;  
  }
  return(solve(sxx) %*% sxy);
}  

# Variance-covariance of the scores;

Omega <- function(beta, Ddata, P, Tp)
{
  x <- Ddata[,-c(1:Tp)];
  y <- Ddata[,c(1:Tp)];  
  N <- nrow(x);
  g <- matrix(0, nrow = Tp*P, ncol = 1);
  g2 <- matrix(0, nrow = Tp*P, ncol = Tp*P);
  
  for (i in 1:nrow(x))
  {
    xi <- matrix(x[i, ], nrow = Tp);
    yi <- matrix(y[i, ], nrow = Tp);
    gi <- matrix(t(kronecker(yi - xi%*%beta, matrix(1, ncol = P)) * xi), ncol=1);
    g  <- g + gi/N; 
    g2 <- g2 + gi %*% t(gi)/N;
  }
  return(g2 - g %*% t(g));
}  

# Gradient of the score function;

Gradient <- function(Ddata, P, Tp)
{
  x <- Ddata[,-c(1:Tp)];
  N <- nrow(x);
  G <- matrix(0, nrow = Tp*P, ncol = P);
  
  for (i in 1:nrow(x))
  {
    xi <- matrix(x[i, ], nrow = Tp);
    Gi <- NULL;
    for (t in 1:Tp)
    {
      Gi <- rbind(Gi, xi[t, ] %*% t(xi[t, ]));
    }
    G  <- G - Gi/N;                
  }
  return(G);
}  

# Over-dentified GMM estimator;

fe.gmm2 <- function(Ddata, P, Tp)
{
  x <- Ddata[,-c(1:Tp)];
  y <- Ddata[,c(1:Tp)];  
  N <- nrow(x);
  sxy <- matrix(0, nrow = P*Tp, ncol = 1);
  Vg  <- Omega(fe.gmm(Ddata, P, Tp), Ddata, P, Tp);
  G   <- Gradient(Ddata, P, Tp);
  for (i in 1:nrow(x))
  {
    xi <- matrix(x[i, ], nrow = Tp);
    yi <- matrix(y[i, ], nrow = Tp);
    sxy <- sxy + matrix(t(kronecker(yi, matrix(1, ncol = P)) * xi), ncol=1)/N;  
  }
  beta <- - solve(t(G) %*% solve(Vg) %*% G) %*% t(G) %*% solve(Vg) %*% sxy;
  return(beta);
}  

# J-test for overidentification;

Jtest <- function(beta, Ddata, P, Tp)
{
  x <- Ddata[,-c(1:Tp)];
  y <- Ddata[,c(1:Tp)];  
  N <- nrow(x);
  g <- matrix(0, nrow = Tp*P, ncol = 1);
  
  for (i in 1:nrow(x))
  {
    xi <- matrix(x[i, ], nrow = Tp);
    yi <- matrix(y[i, ], nrow = Tp);
    gi <- matrix(t(kronecker(yi - xi%*%beta, matrix(1, ncol = P)) * xi), ncol=1);
    g  <- g + gi/N; 
  }
  return(N * t(g) %*% solve(Omega(beta, Ddata, P, Tp)) %*% g);
}  

# 2 step FE-GMM with contemporaneous moment conditions;

fe.gmm.wrapper <- function(data)
{
  Tf <- length(unique(data$year));
  N <- length(unique(data$distid));
  P <- 4; # number of regressors excluding individual and time effects

# Removing individual and time effects from all the variables;

  Dmath4     <- t(matrix(lm(math4 ~ factor(year) + factor(distid), data = data)$res, nrow=Tf));
  Dlrexpp    <- t(matrix(lm(log(rexpp) ~ factor(year) + factor(distid), data = data)$res, nrow=Tf));
  Dll1rexpp  <- t(matrix(lm(log(l1rexpp) ~ factor(year) + factor(distid), data = data)$res, nrow=Tf));
  Dlenrol    <- t(matrix(lm(log(enrol) ~ factor(year) + factor(distid), data = data)$res, nrow=Tf));
  Dlunch     <- t(matrix(lm(lunch ~ factor(year) + factor(distid), data = data)$res, nrow=Tf));

  Ddata <- cbind(Dmath4, Dlrexpp, Dll1rexpp, Dlenrol, Dlunch);

  coefs.gmm <- fe.gmm2(Ddata, P, Tf);
  Jtest.gmm <- Jtest(coefs.gmm, Ddata, P, Tf);

  return(list(coefs = coefs.gmm, J = Jtest.gmm, pJ = 1 - pchisq(Jtest.gmm, P*Tf-P), 
              se = sqrt(diag(solve(t(Gradient(Ddata,P,Tf)) %*% solve(Omega(coefs.gmm, Ddata, P, Tf)) %*% Gradient(Ddata,P,Tf)))/N) ));
}


# 2 step FD-GMM with contemporaneous moment conditions;

fd.gmm.wrapper <- function(data)
{
  Td <- length(unique(data$year)) - 1; # we lost first year with the differencing
  N <- length(unique(data$distid));
  P <- 4; # number of regressors excluding individual and time effects
  
# Remove time effects from differenced variables;

  Dmath4     <- t(matrix(lm(diff(data$math4) ~ factor(year), data = data)$res, nrow=Td));
  Dlrexpp    <- t(matrix(lm(diff(log(data$rexpp)) ~ factor(year), data = data)$res, nrow=Td));
  Dll1rexpp  <- t(matrix(lm(diff(log(data$l1rexpp)) ~ factor(year), data = data)$res, nrow=Td));
  Dlenrol    <- t(matrix(lm(diff(log(data$enrol)) ~ factor(year), data = data)$res, nrow=Td));
  Dlunch     <- t(matrix(lm(diff(data$lunch) ~ factor(year), data = data)$res, nrow=Td));

  Ddata <- cbind(Dmath4, Dlrexpp, Dll1rexpp, Dlenrol, Dlunch);

  coefs.gmm <- fe.gmm2(Ddata, P, Td);
  Jtest.gmm <- Jtest(coefs.gmm, Ddata, P, Td);

  return(list(coefs = coefs.gmm, J = Jtest.gmm, pJ = 1 - pchisq(Jtest.gmm, P*Td-P), 
            se = sqrt(diag(solve(t(Gradient(Ddata,P,Td)) %*% solve(Omega(coefs.gmm, Ddata, P, Td)) %*% Gradient(Ddata,P,Td)))/N) ));

}


# Estimation;

gmm.fe.short.fit  <- fe.gmm.wrapper(rdata);
coefs.fe.short    <- gmm.fe.short.fit$coefs;
Jtest.fe.short    <- gmm.fe.short.fit$J;
pJtest.fe.short   <- gmm.fe.short.fit$pJ;
cse.fe.short  <- gmm.fe.short.fit$se;

gmm.fd.short.fit  <- fd.gmm.wrapper(rdata);
coefs.fd.short    <- gmm.fd.short.fit$coefs;
Jtest.fd.short    <- gmm.fd.short.fit$J;
pJtest.fd.short   <- gmm.fd.short.fit$pJ;
cse.fd.short  <- gmm.fd.short.fit$se;


# Panel bootstrap std errors

set.seed(8);
N <- length(unique(rdata$distid));
T <- length(unique(rdata$year));
R <- 500;
coefs.fe.b  <- matrix(0, ncol = length(coefs.fe.short), nrow = R);
coefs.fd.b  <- matrix(0, ncol = length(coefs.fd.short), nrow = R);

for (r in 1:R) {;
                ids                <- kronecker(sample.int(N, N, replace = TRUE), rep(1,T));
                data.b             <- rdata[(ids-1)*T + rep(c(1:T),N), ];
                data.b$distid      <- kronecker(c(1:N), rep(1,T));
                data.b$year        <- rep(c(1993:1998),N);
                data.b             <- data.frame(data.b);
                data.b             <- pdata.frame(data.b, index = c("distid","year"));         # reset indexes of the panel        
                coefs.fe.b[r, ]    <- fe.gmm.wrapper(data.b)$coefs;
                coefs.fd.b[r, ]    <- fd.gmm.wrapper(data.b)$coefs;  
};

bse.fe.short  <- apply(coefs.fe.b, 2, sd);
bse.fd.short  <- apply(coefs.fd.b, 2, sd);


#### GMM 2: USING ALL MOMENT CONDITIONS FROM STRICT EXOGENEITY

# We use generalized inverses instead of regular inverses because some of the covariance matrices are singular

library(MASS); # package for generalized inverse


# Functions to implement GMM estimators;


fe.gmm <- function(Ddata, P, Tp)
{
  x <- Ddata[,-c(1:Tp)];
  y <- Ddata[,c(1:Tp)];  
  N <- nrow(x);
  sxx <- matrix(0, nrow = P, ncol = P);
  sxy <- matrix(0, nrow = P, ncol = 1);
  
  for (i in 1:nrow(x))
  {
    xi <- matrix(x[i, ], nrow = Tp);
    yi <- matrix(y[i, ], nrow = Tp);
    sxx <- sxx + t(xi) %*% xi;
    sxy <- sxy + t(xi) %*% yi;  
  }
  return(solve(sxx) %*% sxy);
}  

# Variance-covariance of the scores;

Omega <- function(beta, Ddata, Zdata, P, Tp)
{
  x <- Ddata[,-c(1:Tp)];
  y <- Ddata[,c(1:Tp)];  
  N <- nrow(x);
  M <- ncol(Zdata)*Tp; 
  g <- matrix(0, nrow = M, ncol = 1);
  g2 <- matrix(0, nrow = M, ncol = M);
  
  for (i in 1:nrow(x))
  {
    xi <- matrix(x[i, ], nrow = Tp);
    yi <- matrix(y[i, ], nrow = Tp);
    zi <- matrix(Zdata[i, ], ncol = 1);    
    gi <- kronecker(zi, yi - xi%*%beta);
    g  <- g + gi/N; 
    g2 <- g2 + gi %*% t(gi)/N;
  }
  return(g2 - g %*% t(g));
}  

# Gradient of the score function;

Gradient <- function(Ddata, Zdata, P, Tp)
{
  x <- Ddata[,-c(1:Tp)];
  N <- nrow(x);
  M <- ncol(Zdata)*Tp;   
  G <- matrix(0, nrow = M, ncol = P);
  
  for (i in 1:nrow(x))
  {
    xi <- matrix(x[i, ], nrow = Tp);
    zi <- matrix(Zdata[i, ], ncol = 1);    
    Gi <- kronecker(zi, xi);
    G  <- G - Gi/N;                
  }
  return(G);
}  

# Over-dentified GMM estimator;

fe.gmm2 <- function(Ddata, Zdata, P, Tp)
{
  x <- Ddata[,-c(1:Tp)];
  y <- Ddata[,c(1:Tp)];  
  N <- nrow(x);
  M <- ncol(Zdata)*Tp;   
  sxy <- matrix(0, nrow = M, ncol = 1);
  Vg  <- Omega(fe.gmm(Ddata, P, Tp), Ddata, Zdata, P, Tp);
  G   <- Gradient(Ddata, Zdata, P, Tp);
  for (i in 1:nrow(x))
  {
#    xi <- matrix(x[i, ], nrow = Tp);
    zi <- matrix(Zdata[i, ], ncol = 1);    
    yi <- matrix(y[i, ], nrow = Tp);
    sxy <- sxy + kronecker(zi, yi)/N;  
  }
  beta <- - ginv(t(G) %*% ginv(Vg) %*% G) %*% t(G) %*% ginv(Vg) %*% sxy;
  return(beta);
}  

# J-test for overidentification;

Jtest <- function(beta, Ddata, Zdata, P, Tp)
{
  x <- Ddata[,-c(1:Tp)];
  y <- Ddata[,c(1:Tp)];  
  N <- nrow(x);
  M <- ncol(Zdata)*Tp;   
  g <- matrix(0, nrow = M, ncol = 1);
  
  for (i in 1:nrow(x))
  {
    xi <- matrix(x[i, ], nrow = Tp);
    yi <- matrix(y[i, ], nrow = Tp);
    zi <- matrix(Zdata[i, ], ncol = 1);    
    gi <- kronecker(zi, yi - xi%*%beta);
    g  <- g + gi/N; 
  }
  return(list(Jstat = N * t(g) %*% ginv(Omega(beta, Ddata, Zdata, P, Tp)) %*% g, dof = M - P));
}  

# 2 step FE-GMM with strict exogeneity moment conditions;

fe.gmm.wrapper <- function(data)
{
  Tf <- length(unique(data$year));
  N <- length(unique(data$distid));
  P <- 4; # number of regressors excluding individual and time effects
  
  # Removing individual and time effects from all the variables;
  
  Dmath4     <- t(matrix(lm(math4 ~ factor(year) + factor(distid), data = data, subset = (year != 1992))$res, nrow=Tf-1));
  Dlrexpp    <- t(matrix(lm(log(rexpp) ~ factor(year) + factor(distid), data = data, subset = (year != 1992))$res, nrow=Tf-1));
  Dll1rexpp  <- t(matrix(lm(log(l1rexpp) ~ factor(year) + factor(distid), data = data, subset = (year != 1992))$res, nrow=Tf-1));
  Dlenrol    <- t(matrix(lm(log(enrol) ~ factor(year) + factor(distid), data = data, subset = (year != 1992))$res, nrow=Tf-1));
  Dlunch     <- t(matrix(lm(lunch ~ factor(year) + factor(distid), data = data, subset = (year != 1992))$res, nrow=Tf-1));
    
  Ddata <- cbind(Dmath4, Dlrexpp, Dll1rexpp, Dlenrol, Dlunch);
  
  # Removing time effects from the instruments;
  
  Zlrexpp    <- t(matrix(lm(log(rexpp) ~ factor(year), data = data)$res, nrow=Tf));
#  Zll1rexpp  <- t(matrix(lm(log(l1rexpp) ~ factor(year), data = data)$res, nrow=Tf));
  Zlenrol    <- t(matrix(lm(log(enrol) ~ factor(year), data = data)$res, nrow=Tf));
  Zlunch     <- t(matrix(lm(lunch ~ factor(year), data = data)$res, nrow=Tf));
  
  Zdata <- cbind(Zlrexpp, Zlenrol, Zlunch);
  
  
  coefs.gmm <- fe.gmm2(Ddata, Zdata, P, Tf-1);
  Jtest.gmm <- Jtest(coefs.gmm, Ddata, Zdata, P, Tf-1);
  
  return(list(coefs = coefs.gmm, J = Jtest.gmm$Jstat, Jdof = Jtest.gmm$dof, 
              se = sqrt(diag(ginv(t(Gradient(Ddata,Zdata,P,Tf-1)) %*% ginv(Omega(coefs.gmm, Ddata,Zdata, P, Tf-1)) %*% Gradient(Ddata,Zdata,P,Tf-1)))/N) ));
}


# 2 step FD-GMM with strict exogeneity moment conditions;

fd.gmm.wrapper <- function(data)
{
  Tf <- length(unique(data$year)); 
  N <- length(unique(data$distid));
  P <- 4; # number of regressors excluding individual and time effects
  
  # Remove time effects from differenced variables and replace first two years observations for zeros;
  
  Dmath4     <- t(matrix(lm(diff(data$math4) ~ factor(year), data = data, subset = (year != 1993))$res, nrow=Tf-2));
  Dlrexpp    <- t(matrix(lm(diff(log(data$rexpp)) ~ factor(year), data = data, subset = (year != 1993))$res, nrow=Tf-2));
  Dll1rexpp  <- t(matrix(lm(diff(log(data$l1rexpp)) ~ factor(year), data = data, subset = (year != 1993))$res, nrow=Tf-2));
  Dlenrol    <- t(matrix(lm(diff(log(data$enrol)) ~ factor(year), data = data, subset = (year != 1993))$res, nrow=Tf-2));
  Dlunch     <- t(matrix(lm(diff(data$lunch) ~ factor(year), data = data, subset = (year != 1993))$res, nrow=Tf-2));
  
  Ddata <- cbind(Dmath4, Dlrexpp, Dll1rexpp, Dlenrol, Dlunch);
  
  # Removing time effects from the instruments;
  
  Zlrexpp    <- t(matrix(lm(log(rexpp) ~ factor(year), data = data)$res, nrow=Tf));
  Zlenrol    <- t(matrix(lm(log(enrol) ~ factor(year), data = data)$res, nrow=Tf));
  Zlunch     <- t(matrix(lm(lunch ~ factor(year), data = data)$res, nrow=Tf));
  
  Zdata <- cbind(Zlrexpp, Zlenrol, Zlunch);
  
  coefs.gmm <- fe.gmm2(Ddata, Zdata, P, Tf-2);
  Jtest.gmm <- Jtest(coefs.gmm, Ddata, Zdata, P, Tf-2);
  
  return(list(coefs = coefs.gmm, J = Jtest.gmm$Jstat, Jdof = Jtest.gmm$dof, 
              se = sqrt(diag(ginv(t(Gradient(Ddata,Zdata,P,Tf-2)) %*% ginv(Omega(coefs.gmm, Ddata,Zdata, P, Tf-2)) %*% Gradient(Ddata,Zdata,P,Tf-2)))/N) ));
    
}


# Estimation;

gmm.fe.long.fit  <- fe.gmm.wrapper(data);
coefs.fe.long    <- gmm.fe.long.fit$coefs;
Jtest.fe.long    <- gmm.fe.long.fit$J;
Jdof.fe.long   <- gmm.fe.long.fit$Jdof;
cse.fe.long  <- gmm.fe.long.fit$se;

gmm.fd.long.fit  <- fd.gmm.wrapper(data);
coefs.fd.long    <- gmm.fd.long.fit$coefs;
Jtest.fd.long    <- gmm.fd.long.fit$J;
Jdof.fd.long   <- gmm.fd.long.fit$Jdof;
cse.fd.long  <- gmm.fd.long.fit$se;


# Panel bootstrap std errors

set.seed(8);
N <- length(unique(data$distid));
T <- length(unique(data$year));
R <- 500;
coefs.fe.b  <- matrix(0, ncol = length(coefs.fe.long), nrow = R);
coefs.fd.b  <- matrix(0, ncol = length(coefs.fd.long), nrow = R);

for (r in 1:R) {;
                ids                <- kronecker(sample.int(N, N, replace = TRUE), rep(1,T));
                data.b             <- data[(ids-1)*T + rep(c(1:T),N), ];
                data.b$distid      <- kronecker(c(1:N), rep(1,T));
                data.b$year        <- rep(c(1992:1998),N);
                data.b             <- data.frame(data.b);
                data.b             <- pdata.frame(data.b, index = c("distid","year"));         # reset indexes of the panel        
                coefs.fe.b[r, ]    <- fe.gmm.wrapper(data.b)$coefs;
                coefs.fd.b[r, ]    <- fd.gmm.wrapper(data.b)$coefs;  
};

bse.fe.long  <- apply(coefs.fe.b, 2, sd);
bse.fd.long  <- apply(coefs.fd.b, 2, sd);


######## Table of results;


table.all <- matrix(NA, nrow = 15, ncol = 7, dimnames = list(c("log(rexpp)", "CSE", "BSE", "log(l1.rexpp)",  "CSE1", "BSE1", "log(enrol)",  "CSE2", "BSE2","lunch",  "CSE3", "BSE3","J-test", "p-val", "d.o.f."), c("Pooled", "FD", "GMM-FD", "GMM-FD",  "FE", "GMM-FE", "GMM-FE")));

table.all[c(1,4,7,10), 1] <- coefs.ols[2:5];
table.all[c(2,5,8,11), 1] <- cse.ols[2:5];
table.all[c(3,6,9,12), 1] <- bse.ols[2:5];

table.all[c(1,4,7,10), 2] <- coefs.fd[2:5];
table.all[c(2,5,8,11), 2] <- cse.fd[2:5];
table.all[c(3,6,9,12), 2] <- bse.fd[2:5];

table.all[c(1,4,7,10), 5] <- coefs.fe[1:4];
table.all[c(2,5,8,11), 5] <- cse.fe[1:4];
table.all[c(3,6,9,12), 5] <- bse.fe[1:4];

table.all[c(1,4,7,10), 3] <- coefs.fd.short;
table.all[c(2,5,8,11), 3] <- cse.fd.short;
table.all[c(3,6,9,12), 3] <- bse.fd.short;

table.all[c(1,4,7,10), 6] <- coefs.fe.short;
table.all[c(2,5,8,11), 6] <- cse.fe.short;
table.all[c(3,6,9,12), 6] <- bse.fe.short;

table.all[c(1,4,7,10), 4] <- coefs.fd.long;
table.all[c(2,5,8,11), 4] <- cse.fd.long;
table.all[c(3,6,9,12), 4] <- bse.fd.long;

table.all[c(1,4,7,10), 7] <- coefs.fe.long;
table.all[c(2,5,8,11), 7] <- cse.fe.long;
table.all[c(3,6,9,12), 7] <- bse.fe.long;

table.all[13 ,3] <- Jtest.fd.short;
table.all[14 ,3] <- pJtest.fd.short;

table.all[13 ,6] <- Jtest.fe.short;
table.all[14 ,6] <- pJtest.fe.short;

table.all[13 ,4] <- Jtest.fd.long;
table.all[14 ,4] <- 1 - pchisq(Jtest.fd.long, Jdof.fd.long);

table.all[13 ,7] <- Jtest.fe.long;
table.all[14 ,7] <- 1 - pchisq(Jtest.fe.long, Jdof.fe.long);

P <- 4; # number of regressors excluding individual and time effects


table.all[15 ,3] <- P*(T-2)-P ;
table.all[15 ,6] <- P*(T-1)-P ;
table.all[15 ,4] <- Jdof.fd.long ;
table.all[15 ,7] <- Jdof.fe.long ;


xtable(table.all);


