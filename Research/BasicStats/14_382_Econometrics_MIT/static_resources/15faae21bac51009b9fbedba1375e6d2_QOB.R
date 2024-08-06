
# This empirical example uses the data from Angrist and Krueger (1991) to illustrate the construction of 
# confidence intervals robust to weak instruments using the method of Chernozhukov and Hansen (2008)

# Authors: V. Chernozhukov and I. Fernandez-Val

# Data source: Josh Angrist and Alan Krueger, "Does compulsory school attendance affect schooling and earnings?", 
# Quarterly Journal of Economics, Vol. 106, No. 4, 1991, pp. 979-1014
# URL for data: http://economics.mit.edu/faculty/angrist/data1/data/angkru1991

# Updated on 02/16/2015

############

# Part I: Functions

# Function to compute White standard errors;

hc <- function(x) 
{;
  vcovHC(x, type = "HC");
};

####################

# Part II: Main program

library(sandwich);
library(foreign);
library(AER);
#library(ivpack);

# Sample: Census 1980, Cohort 1930-1939;

data <- read.dta("/Users/ivan/Dropbox/Shared/14.382/Data/NEW7080/QOB-CENSUS80-COHORT30-39.dta");
attach(data);
alpha <- .05; # Significance level

# Partialing-out controls;

rlwage <- lm(LWKLYWGE ~ factor(YOB) +  factor(MARRIED) +  factor(RACE) + factor(SMSA) + factor(REGION))$res;
reduc  <- lm(EDUC ~ factor(YOB) +  factor(MARRIED) +  factor(RACE) + factor(SMSA) + factor(REGION))$res;
rqob4  <- lm(I(QOB == 4) ~ factor(YOB) +  factor(MARRIED) +  factor(RACE) + factor(SMSA) + factor(REGION))$res;
  
# ols results;

formula  <- rlwage ~ reduc;
ols.fit  <- lm(formula);
ols.coef <- ols.fit$coef[2];
ols.se   <- coeftest(ols.fit, vcov = hc)[2,2];
ols.lci  <- ols.coef + qnorm(alpha/2)*ols.se;
ols.uci  <- ols.coef + qnorm(1-alpha/2)*ols.se;


# First stage;

formula   <- reduc ~ rqob4;
fs.fit   <- lm(formula);
fs.coef  <- fs.fit$coef[2];
fs.se    <- coeftest(fs.fit, vcov = hc)[2,2];
fs.Fstat <- (fs.coef/fs.se)^2;
print(paste('F-stat: ', fs.Fstat));



# tsls results

formula   <- rlwage ~ reduc | rqob4;

tsls.fit  <- ivreg(formula);
tsls.coef <- tsls.fit$coef[2];
tsls.se   <- coeftest(tsls.fit, vcov = hc)[2,2]; 
tsls.lci  <- tsls.coef + qnorm(alpha/2)*tsls.se;
tsls.uci  <- tsls.coef + qnorm(1-alpha/2)*tsls.se;


# Confidence interval robust to weak instruments (Chernozhukov and Hansen, 2008)

formula <- I(rlwage - beta * reduc) ~ rqob4;

gridbeta  <- tsls.coef + c(-50:50)/750;
gridbeta  <- tsls.coef + c(-100:100)/1500;
gridrtstat <- 0*gridbeta;


for (i in 1:length(gridbeta)) {;
  beta  <- gridbeta[i];                             
  fit   <- lm(formula);
  gridrtstat[i] <- summary(fit)$coef[2,1]/coeftest(fit, vcov = hc)[2,2];
  };

tsls.lrci <- min(gridbeta[gridrtstat^2 < qchisq(1-alpha,1)]);
tsls.urci <- max(gridbeta[gridrtstat^2 < qchisq(1-alpha,1)]); 


options(digits=3);

table <- matrix(0, ncol = 4, nrow = 3, dimnames = list(c('OLS', 'TSLS - Wald', 'WI Robust'), c('Est.', 'Std. Error', '95% LCI','95% UCI')));

table[1,1] <- ols.coef;
table[1,2] <- ols.se;
table[1,3] <- ols.lci;
table[1,4] <- ols.uci;

table[2,1] <- tsls.coef;
table[2,2] <- tsls.se;
table[2,3] <- tsls.lci;
table[2,4] <- tsls.uci;

table[3,1] <- NA;
table[3,2] <- NA;
table[3,3] <- tsls.lrci;
table[3,4] <- tsls.urci;

print(table);

# Graphical illustration of the construction of the confidence intervals;

tstat.tsls <- ((tsls.coef - gridbeta)/tsls.se)^2;
tstat.wi   <- gridrtstat^2;

  
pdf("/Users/ivan/Dropbox/Shared/14.382/Results/QOB.pdf", pointsize=15,width=8.0,height=8.0);

par(mfrow=c(1,1))

  plot(range(gridbeta),range(c(tstat.tsls, tstat.wi)) , type="n",xlab="Returns to Schooling", ylab="Statistic", main=" ");
  lines(gridbeta, tstat.tsls, lty = 2, col = 2);     
  lines(gridbeta, tstat.wi,   lty = 1, col = 1);       
  abline(h=qchisq(1-alpha,1), lty = 3, col = 4);

  legend(min(gridbeta), max(tstat.tsls), c('Wald Statistic', 'WI-Robust Statistic','95% Critical Value'), col = c(2,1,4), lty = c(2,1,3), horiz = F, bty = 'n');

dev.off()

detach(data);

