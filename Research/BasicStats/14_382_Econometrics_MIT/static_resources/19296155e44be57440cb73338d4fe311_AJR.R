
# This empirical example uses the data from Acemoglu, Johnson and Robinson (2001) to illustrate the construction of 
# confidence intervals robust to weak instruments using the method of Chernozhukov and Hansen (2008)

# Authors: V. Chernozhukov and I. Fernandez-Val

# Data source: Acemoglu, D., Johnson, S., Robinson, J.A., 2001. The Colonial Origins of Comparative Development: An 
# Empirical Investigation. American Economic Review 91 (5), 1369–1401
# URL for data: http://faculty.chicagobooth.edu/christian.hansen/research/

# Updated on 02/22/2015

############

# Part I: Functions

# Function to compute White standard errors with small sample adjustment;

hc <- function(x) 
{;
 vcovHC(x, type = "HC3");
};

####################

# Part II: Main program

library(sandwich);
library(foreign);
library(AER);
#library(ivpack);

# Reading the data;

data <- as.data.frame(read.table("/Users/ivan/Dropbox/Shared/14.382/Data/AJR/acemoglu_col.txt", header=T ));
attach(data);
alpha <- .05; # Significance level

# In this example it is more convenient not to partial-out the controls because the sample size is small. If you partial out, 
# you would need to manually correct the small sample adjustments of the  White standard errors

# ols results;

formula  <- GDP ~ Exprop + Latitude;
ols.fit  <- lm(formula);
ols.coef <- ols.fit$coef[2];
ols.se   <- coeftest(ols.fit, vcov = hc)[2,2];
ols.lci  <- ols.coef + qnorm(alpha/2)*ols.se;
ols.uci  <- ols.coef + qnorm(1-alpha/2)*ols.se;


# First stage;

formula   <- Exprop ~ log(Mort) + Latitude;
fs.fit   <- lm(formula);
fs.coef  <- fs.fit$coef[2];
fs.se    <- coeftest(fs.fit, vcov = hc)[2,2];
fs.Fstat <- (fs.coef/fs.se)^2;
print(paste('F-stat: ', fs.Fstat));


# tsls results

formula   <- GDP ~ Exprop + Latitude| log(Mort) + Latitude;

tsls.fit  <- ivreg(formula);
tsls.coef <- tsls.fit$coef[2];
tsls.se   <- coeftest(tsls.fit, vcov = hc)[2,2]; 
tsls.lci  <- tsls.coef + qnorm(alpha/2)*tsls.se;
tsls.uci  <- tsls.coef + qnorm(1-alpha/2)*tsls.se;


# Confidence interval robust to weak instruments (Chernozhukov and Hansen, 2008)

formula <- I(GDP - beta * Exprop) ~ log(Mort) + Latitude;

gridbeta  <- tsls.coef + c(-100:150)/25 * tsls.se;
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

tstat.tsls <- ((tsls.coef - gridbeta)/tsls.se)^2;
tstat.wi   <- gridrtstat^2;


postscript("/Users/ivan/Dropbox/Shared/14.382/Results/AJR.eps",horizontal=F, pointsize=15,width=8.0,height=8.0)

par(mfrow=c(1,1))

plot(range(gridbeta),range(c(tstat.tsls, tstat.wi)) , type="n",xlab="Effect of institutions", ylab="Statistic", main=" ");
lines(gridbeta, tstat.tsls, lty = 2, col = 2);     
lines(gridbeta, tstat.wi,   lty = 1, col = 1);       
abline(h=qchisq(1-alpha,1), lty = 3, col = 4);

legend(min(gridbeta), max(tstat.wi), c('Wald Statistic', 'WI-Robust Statistic','95% Critical Value'), col = c(2,1,4), lty = c(2,1,3), horiz = F, bty = 'n');

dev.off()

detach(data);


