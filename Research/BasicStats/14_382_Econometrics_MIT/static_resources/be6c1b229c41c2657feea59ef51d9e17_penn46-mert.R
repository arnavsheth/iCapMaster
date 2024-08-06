# This empirical example uses the data from Pennsylvania Reemployment Bonus Experiments to illustrate the construction of 
# simultaneous confidence bands

# Authors: V. Chernozhukov and I. Fernandez-Val

# Data source: Yannis Bilias, "Sequential Testing of Duration Data: The Case of Pennsylvania 'Reemployment Bonus' Experiment", 
# Journal of Applied Econometrics, Vol. 15, No. 6, 2000, pp. 575-594

# Description of the data set taken from Bilias (2000):

# "Our extract has 13913 observations on 23 variables (some of which are
#                                                     dummies constructed from the original definitions.) These data are in the
# file penn_jae.dat, which is an ASCII file in DOS format that is zipped in
# bilias-data.zip.
# 
# The 23 variables (columns) of the datafile utilized in the article
# may be described as follows:
#   
#   abdt:  chronological time of enrollment of each claimant
# in the Pennsylvania reemployment bonus experiment.
# 
# tg:  indicates the treatment group (bonus amount - qualification period)
# of each claimant. 
# if tg=0, then claimant enrolled in the control group
# if tg=1, then claimant enrolled in the group 1, and so on.
# (for definitions of the each group see the article, or the 
#  Final Report).
# 
# inuidur1: a measure of length (in weeks) of the first spell of 
# unemployment; this measure was used in the empirical 
# analysis of this article.
# (this is a constructed variable and 
#  the following is a quote from the documentation:
#    "This variable reflected the number of weeks in the claimant's
#  initial UI duration using a break-in-payments definition of a spell.
#  If a claimant did not collect any weeks of UC, INUIDUR1 was set to 1
#  because he/she must have signed for at least a waiting week in order
#  to have been selected for the demonstration.  If a claimant had a gap
#  of at least 7 weeks between the AB_DT and the first claim week paid,
#  INUIDUR1 was also set to 1 to capture the waiting week.  Otherwise,
#  the initial UI duration was deemed to have ended if there was a break
#  in payments of at least 3 weeks' duration.  In this instance, INUIDUR1
#  was set equal to the duration of the spell up to the break, plus one
#  for the waiting week.  For all other cases, INUIDUR1 equalled the
#  length of the spell plus one for the waiting week."
#  
#  inuidur2: a second measure for the length (in weeks) of 
#  the first spell of unemployment;
#  it was not used in our data analysis.
#  
#  female: dummy variable; it indicates if the claimant's sex 
#  is female (=1) or male (=0).
#  
#  black: dummy variable; it  indicates a person of black race (=1).
#  
#  hispanic: dummy variable; it  indicates a person of hispanic race (=1).
#  
#  othrace: dummy variable; it  indicates a non-white, non-black, not-hispanic 
#  person (=1).
#  
#  dep:  the number of dependents of each claimant;
#  In case the claimant has 2 or more dependents,
#  it is equal to 2.  Else it is 0 or 1 accordingly.
#  
#  q1-q6: six dummy variables indicating the quarter of experiment
#  during which each claimant enrolled.
#  
#  recall: takes the value of 1 if the claimant answered ``yes'' when
#  was asked if he/she had any expectation to be recalled.
#  
#  agelt35: takes the value of 1 if the claimant's age is less
#  than 35 and 0 otherwise.
#  
#  agegt54: takes the value of 1 if the claimant's age is more
#          than 54 and 0 otherwise.
# 
# durable: it takes the value of 1 if the occupation
#          of the claimant was in the sector of durable manufacturing
#          and 0 otherwise.
# 
# nondurable: it takes the value of 1 if the occupation
#             of the claimant was in the sector of nondurable 
#             manufacturing and 0 otherwise.
# 
# lusd: it takes the value of 1 if the claimant filed
#       in Coatesville, Reading, or Lancaster and 0 otherwise.
#       These three sites were considered to be located
#       in areas characterized by low unemployment rate and
#       short duration of unemployment.
# 
# husd: it takes the value of 1 if the claimant filed
#       in Lewistown, Pittston, or Scranton and 0 otherwise.
#       These three sites were considered to be located
#       in areas characterized by high unemployment rate and
#       short duration of unemployment.
# 
# muld: it takes the value of 1 if the claimant filed
#       in Philadelphia-North, Philadelphia-Uptown, McKeesport, 
#       Erie, or Butler and 0 otherwise.
#       These three sites were considered to be located
#       in areas characterized by moderate unemployment rate and
#       long duration of unemployment."


# Updated on 02/6/2015

############

# Part I: Functions

# This function obtains quantiles of the maximal t-statistic by simulation

qtmax <- function(C, S, alpha)
  {;
   p <- nrow(C);
   tmaxs <- apply(abs(msqrt(C) %*% matrix(rnorm(p*S), nrow = p, ncol = S)), 2, max);
   return(quantile(tmaxs, 1-alpha));
  };

# This function computes the square root of a symmetric matrix using the spectral decomposition;

msqrt <- function(C)
  {;
  C.eig <- eigen(C);
  return(C.eig$vectors %*% diag(sqrt(C.eig$values)) %*% solve(C.eig$vectors));
  };

####################

# Part II: Main program

library(sandwich);

filepathIvan<-"/Users/Ivan/Dropbox/Shared/14.382"
filepathVictor<-"/Users/VC/Dropbox/TEACHING/14.382"
filepathMert<-"/Users/mertdemirer1/Dropbox (Personal)/14.382"
filepathVira<-"/Users/virasemenova/Dropbox/14.382"
setwd(filepathVictor)

Penn<- as.data.frame(read.table("Data/penn_jae.dat", header=T ));

nn<- dim(Penn)[1];
attach(Penn);
alpha <- .1; # Significance level
S <- 100000; # Number of simulations to estimate the critical value 
set.seed(888);


#ols results

tg[tg == 6]   <- 4; 
formula <- log(inuidur1)~factor(tg)+female+black+othrace+factor(dep)+q2+q3+q4+q5+q6+agelt35+agegt54+durable+lusd+husd;
# Omitted dummies: q1, nondurable, muld;

ols.fit <- lm(formula);
coefs   <- coef(ols.fit);
vars <- names(coefs);
HCV.coefs <- vcovHC(ols.fit, type = 'HC');
coefs.se <- sqrt(diag(HCV.coefs)); # White std errors
C.coefs  <- (diag(1/sqrt(diag(HCV.coefs)))) %*% HCV.coefs %*% (diag(1/sqrt(diag(HCV.coefs))));

tes  <- coefs[2:6];
tes.se <- coefs.se[2:6];
tes.lev <- c('T1','T2','T3','T4','T5');
tes.cor <- C.coefs[2:6, 2:6];

crit.val <- qtmax(tes.cor,S,alpha);

print(crit.val);

tes.ucb  <- tes + crit.val * tes.se;
tes.lcb  <- tes - crit.val * tes.se;

tes.uci  <- tes + qnorm(1-alpha/2) * tes.se;
tes.lci  <- tes + qnorm(alpha/2) * tes.se;

postscript("L1/Results/TEs-CB.eps",horizontal=F, pointsize=15,width=8.0,height=8.0)

par(mfrow=c(1,1))

plot( c(1,5), las = 2, xlim =c(0.6, 5.4), ylim = c(-.16, .09),  type="n",xlab="Treatment", ylab="Average Effect (log of weeks)", main="Treatment Effects on Unemployment Duration", xaxt="n");
axis(1, at=1:5, labels=tes.lev);
for (i in 1:5)
{;
 rect(i-0.2, tes.lci[i], i+0.2,  tes.uci[i], col = NA,  border = "red", lwd = 3);    
 rect(i-0.2, tes.lcb[i], i+0.2, tes.ucb[i], col = NA,  border = 4, lwd = 3 );   
 segments(i-0.2, tes[i], i+0.2, tes[i], lwd = 5 );
};
abline(h=0);

legend(2.5, 0.1, c('Regression Estimate', '90% Simultaneous Confidence Interval', '90% Pointwise Confidence Interval'), col = c(1,4,2), lwd = c(4,3,3), horiz = F, bty = 'n', cex=0.8);

dev.off()

### build a more flexible specification by taking all the two-way interactions of controls

formula2 <- log(inuidur1)~factor(tg)+(female+black+othrace+factor(dep)+q2+q3+q4+q5+q6+agelt35+agegt54+durable+lusd+husd)^2;
# Omitted dummies: q1, nondurable, muld;

ols.fit <- lm(formula2);
coefs   <- coef(ols.fit);
vars <- names(coefs);
HCV.coefs <- vcovHC(ols.fit, type = 'HC');
coefs.se <- sqrt(diag(HCV.coefs)); # White std errors
C.coefs  <- (diag(1/sqrt(diag(HCV.coefs)))) %*% HCV.coefs %*% (diag(1/sqrt(diag(HCV.coefs))));

tes  <- coefs[2:6];
tes.se <- coefs.se[2:6];
tes.lev <- c('T1','T2','T3','T4','T5');
tes.cor <- C.coefs[2:6, 2:6];

crit.val <- qtmax(tes.cor,S,alpha);

print(crit.val);

tes.ucb  <- tes + crit.val * tes.se;
tes.lcb  <- tes - crit.val * tes.se;

tes.uci  <- tes + qnorm(1-alpha/2) * tes.se;
tes.lci  <- tes + qnorm(alpha/2) * tes.se;

postscript("L1/Results/TEs-CB-flex.eps",horizontal=F, pointsize=15,width=8.0,height=8.0)

par(mfrow=c(1,1))

plot( c(1,5), las = 2, xlim =c(0.6, 5.4), ylim = c(-.16, .09),  type="n",xlab="Treatment", ylab="Average Effect (log of weeks)", main="Treatment Effects on Unemployment Duration", xaxt="n");
axis(1, at=1:5, labels=tes.lev);
for (i in 1:5)
{;
 rect(i-0.2, tes.lci[i], i+0.2,  tes.uci[i], col = NA,  border = "red", lwd = 3);    
 rect(i-0.2, tes.lcb[i], i+0.2, tes.ucb[i], col = NA,  border = 4, lwd = 3 );   
 segments(i-0.2, tes[i], i+0.2, tes[i], lwd = 5 );
};
abline(h=0);

legend(2.5, 0.1, c('Regression Estimate', '90% Simultaneous Confidence Interval', '90% Pointwise Confidence Interval'), col = c(1,4,2), lwd = c(4,3,3), horiz = F, bty = 'n', cex=0.8);

dev.off()




