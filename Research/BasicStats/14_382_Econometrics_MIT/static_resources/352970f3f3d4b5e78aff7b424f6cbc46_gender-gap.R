
# This empirical example uses the data from the CPS to illustrate the use of distribution regression and counterfactual distributions
# in the estimation of the gender gap

# Authors: V. Chernozhukov and I. Fernandez-Val

# Data source: IPUMS-CPS through Paul Schrimpf
# URL for data: https://cps.ipums.org/cps/

# Description of the data: the sample selection and variable contruction follow

# Mulligan, Casey B., and Yona Rubinstein. "Selection, investment, and women's relative wages over time." The Quarterly Journal of Economics (2008): 1061-1110.

# Sample selection: white non-hipanic, ages 25-54, working full time full year (35+ hours per week at least 50 weeks), exclude living in group quarters, 
# self-employed, military, agricultural, and private household sector, allocated earning, inconsistent report on earnings and employment, missing data

# The variables used in the analysis include:

# - lnw: log of hourly wage (annual earnings / annual hours)
# - female: female indicator
# - 6 married status indicators: widowed, divorced, separated, nevermarried, and married (omitted)
# - 6 education attainment indicators: hsd08, hsd911, hsg, cg, ad, and sc (omitted)
# - 4 region indicators: mw, so, we, and ne (omitted)
# - Quartic in potential experience (max[0, age - years of education - 7]): exp1, exp2 (divided by 100), exp3 (divided by 1000), exp4 (divided by 10000)
# - weight: March Supplement sampling weight
# - year: CPS year

# Updated on 03/25/2015

############

# Part I: Functions


# Fuction to estimate weighted distribution;

wcdf <- function(y, data, group)
{;
 Ff    <- weighted.mean((data$lnw[data$female==group] <= y), data$weight[data$female==group]); 
 return(Ff);
};


# Fuction to estimate weighted distribution in bootstrap;

boot.wcdf <- function(data, indices, ys, group)
{;
 data.b<- data[indices,];  
 Ff    <- sapply(ys, wcdf, data = data.b, group = group);
 return(Ff);
};

# Fuction to estimate weighted distribution using distribution regression;

wcdf2 <- function(y, data)
{;
 form  <- I(lnw < y) ~ widowed + divorced + separated + nevermarried +(hsd08+hsd911+hsg+cg+ad)*(exp1+exp2+exp3+exp4) + mw + so + we; 
 nind  <- length(ys);
 fit   <- glm(form, family = binomial(link = "logit"), subset = (female==1), weight = weight/sum(weight), data = data);
 Ff    <- weighted.mean(predict(fit, type = "response"), data$weight[data$female==1]); 
 return(Ff);
};


# Fuction to estimate weighted distribution using distribution regression in bootstrap;

boot.wcdf2 <- function(data, indices, ys)
{;
 form  <- I(lnw < y) ~ widowed + divorced + separated + nevermarried +(hsd08+hsd911+hsg+cg+ad)*(exp1+exp2+exp3+exp4) + mw + so + we; 
 data.b<- data[indices,];  
 Ff    <- sapply(ys, wcdf2, data = data.b);
 return(Ff);
};



# Fuction to estimate counterfactual distribution using distribution regression;

counter <- function(y, data)
{;
 form  <- I(lnw < y) ~ widowed + divorced + separated + nevermarried +(hsd08+hsd911+hsg+cg+ad)*(exp1+exp2+exp3+exp4) + mw + so + we; 
 nind  <- length(ys);
 fit   <- glm(form, family = binomial(link = "logit"), subset = (female==0), weight = weight/sum(weight), data = data);
 Ff_c  <- weighted.mean(predict(fit, type = "response", newdata = data[data$female==1, ]), data$weight[data$female==1]); 
 return(Ff_c);
};

# Fuction to estimate counterfactual distribution using distribution regression in bootstrap;

boot.counter <- function(data, indices, ys)
{;
 form  <- I(lnw < y) ~ widowed + divorced + separated + nevermarried +(hsd08+hsd911+hsg+cg+ad)*(exp1+exp2+exp3+exp4) + mw + so + we; 
 data.b<- data[indices,];  
 Ff_c  <- sapply(ys, counter, data = data.b);
 return(Ff_c);
};


# Functions to obtain step-function for distribution;
cdf<-  function(ys, Fs){;
  ys<- sort(ys);
  Fs<- sort(Fs);
  F<- approxfun(ys, Fs, method="constant", f=0, rule=1:2);
  return(F);
};


# Function to obtain step-function for left-inverse (quantile);
left.inv<- function(ys, Fs, rule) {;
  ys<- sort(ys);
  Fs<- sort(Fs);  
  iF<- approxfun(Fs, ys, method="constant", f=1, rule=rule);
  return(iF);
};




####################

# Part II: Main program


# # Reading the souce data;
# 
# load('/Users/Ivan/Dropbox/Shared/14.382/Data/CPS/ipumsCPS1976_2012.rdata');
# 
# # Sample and variable selection;
# 
# ipums$married <- ipums$marst<=2;
# ipums$ne <- ipums$region>=11 & ipums$region<20;
# ipums$sc <- ipums$educ>073 & ipums$educ<110;
# 
# keeps = c("lnw", "female", "widowed", "divorced", "separated", "nevermarried", "married", "hsd08","hsd911","hsg","sc","cg","ad",
#           "mw", "so", "we", "ne",  "exp1", "exp2", "exp3", "exp4", "weight", "year");
# 
# 
# cps <- ipums[ipums$ftfy50 == 1 & !is.na(ipums$lnw), (colnames(ipums) %in% keeps)];
# cps2012 <- cps[cps$year == 2012, ];
# cps2007 <- cps[cps$year == 2007, ];
# cps2002 <- cps[cps$year == 2002, ];
# 
# 
# save(cps, file="/Users/Ivan/Dropbox/Shared/14.382/Data/CPS/cps_all.Rdata");
# save(cps2012, file="/Users/Ivan/Dropbox/Shared/14.382/Data/CPS/cps2012.Rdata");
# save(cps2007, file="/Users/Ivan/Dropbox/Shared/14.382/Data/CPS/cps2007.Rdata");
# save(cps2002, file="/Users/Ivan/Dropbox/Shared/14.382/Data/CPS/cps2002.Rdata");
# 
# rm(ipums, cps, cps2012, cps2007, cps2002, keeps);

# Main Analysis;

filepathIvan<-"/Users/Ivan/Dropbox/Shared/14.382"
filepathVictor<-"/Users/VC/Dropbox/TEACHING/14.382"
filepathMert<-"/Users/mertdemirer1/Dropbox (Personal)/14.382"
filepathVira<-"/Users/virasemenova/Dropbox/14.382"
setwd(filepathMert)

options(warn=-1); # sets warnings off;
load('Data/CPS/cps2012.rdata');
data <- cps2012;
attach(data);

# Estimation of status quo and counterfactual distributions;

nind  <- 39; # Number of distribution regressions to estimate conditional and counterfactual distribution;
ys <- quantile(lnw, c(.02,c(1:nind)/(nind+1), .98));

#nind  <- 99; # Number of distribution regressions to estimate conditional and counterfactual distribution;
#ys <- quantile(lnw, c(1:nind)/(nind+1));

#ys <- sort(unique(lnw));
  
Ff     <- sapply(ys, wcdf, data = data, group = 1);
Ff2    <- sapply(ys, wcdf2, data = data);
Ff_c   <- sort(sapply(ys, counter, data = data));
Fm     <- sapply(ys, wcdf, data = data, group = 0);

#system.time(sapply(ys, wcdf, data = data));
#system.time(sapply(ys, wcdf2, data = data));
#system.time(sapply(ys, counter, data = data));

# Bootstrap confidence intervals;

alpha <- 0.05;
library(boot);

# 1 - status quo distributions;

# Women;
set.seed(1);
wdist.boot   <- boot(data=data, statistic=boot.wcdf, ys = ys, group = 1, R=200, parallel = "multicore", ncpus = 24);

Ff.b <- t(wdist.boot$t); # bootstrap draws

# # centered/scaled draws;
# zs<- apply(abs((Ff.b-Ff)/sqrt(apply(Ff.b, 1, var))), 2, max);   #max abs t-stat;
# crt<- quantile(zs, 1-alpha, na.rm = TRUE)  #critical value
# 
# ubound.Ff<-  sort(Ff + crt*sqrt(apply(Ff.b, 1, var)));  #upper conf band
# lbound.Ff<-  sort(Ff - crt*sqrt(apply(Ff.b, 1, var)));  #lower conf band

# centered/scaled draws
delta    <- Ff.b-Ff;
variance <- apply(delta*delta,1,mean)
zs<- apply(abs(delta) /sqrt(variance),  2, max)  #max abs t-stat
crt<- quantile(zs, 1-alpha)  #critical value

ubound.Ff<-  sort(Ff + crt*sqrt(variance));  #upper conf band
lbound.Ff<-  sort(Ff - crt*sqrt(variance));  #lower conf band




#Men;

set.seed(1);
wdist0.boot   <- boot(data=data, statistic=boot.wcdf, ys = ys, group = 0, R=200, parallel = "multicore", ncpus = 24);

Fm.b <- t(wdist0.boot$t); # bootstrap draws

# # centered/scaled draws;
# zs<- apply(abs((Fm.b-Fm)/sqrt(apply(Fm.b, 1, var))), 2, max);   #max abs t-stat;
# crt<- quantile(zs, 1-alpha, na.rm = TRUE)  #critical value
# 
# ubound.Fm<-  sort(Fm + crt*sqrt(apply(Fm.b, 1, var)));  #upper conf band
# lbound.Fm<-  sort(Fm - crt*sqrt(apply(Fm.b, 1, var)));  #lower conf band

# centered/scaled draws
delta    <- Fm.b-Fm;
variance <- apply(delta*delta,1,mean)
zs<- apply(abs(delta) /sqrt(variance),  2, max)  #max abs t-stat
crt<- quantile(zs, 1-alpha)  #critical value

ubound.Fm<-  sort(Fm + crt*sqrt(variance));  #upper conf band
lbound.Fm<-  sort(Fm - crt*sqrt(variance));  #lower conf band


# 2 - Counterfactual distribution;

set.seed(1);
counter.boot   <- boot(data=data, statistic=boot.counter, ys = ys, R=200, parallel = "multicore", ncpus = 24);

Ff_c.b <- t(counter.boot$t); # bootstrap draws

# # centered/scaled draws;
# zs<- apply(abs((Ff_c.b-Ff_c)/sqrt(apply(Ff_c.b, 1, var))), 2, max);   #max abs t-stat;
# crt_c<- quantile(zs, 1-alpha)  #critical value

# ubound.Ff_c <-  sort(Ff_c + crt_c*sqrt(apply(Ff_c.b, 1, var)));  #rearranged upper conf band
# lbound.Ff_c <-  sort(Ff_c - crt_c*sqrt(apply(Ff_c.b, 1, var)));  #rearranged lower conf band
# 
# ubound.Ff_c <-  ifelse(ubound.Ff_c <= 1, ubound.Ff_c, 1);  #imposing support restriction
# lbound.Ff_c <-  ifelse(lbound.Ff_c >= 0, lbound.Ff_c, 0);  #imposing support restriction


# centered/scaled draws
delta    <- Ff_c.b-Ff_c;
variance <- apply(delta*delta,1,mean)
zs       <- apply(abs(delta) /sqrt(variance),  2, max)  #max abs t-stat
crt_c    <- quantile(zs, 1-alpha)  #critical value

ubound.Ff_c <-  sort(Ff_c + crt_c*sqrt(variance));  #rearranged upper conf band
lbound.Ff_c <-  sort(Ff_c - crt_c*sqrt(variance));  #rearranged lower conf band

ubound.Ff_c <-  ifelse(ubound.Ff_c <= 1, ubound.Ff_c, 1);  #imposing support restriction
lbound.Ff_c <-  ifelse(lbound.Ff_c >= 0, lbound.Ff_c, 0);  #imposing support restriction




# Quantiles, QTEs and confidence bands;

# Quantile functions;
Qf.func  <- approxfun(Ff, ys, method = "linear");
uQf.func <- approxfun(lbound.Ff, ys, method = "linear");
lQf.func <- approxfun(ubound.Ff, ys, method = "linear");

Qm.func  <- approxfun(Fm, ys, method = "linear");
uQm.func <- approxfun(lbound.Fm, ys, method = "linear");
lQm.func <- approxfun(ubound.Fm, ys, method = "linear");

Qf_c.func  <- approxfun(Ff_c, ys, method = "linear");
uQf_c.func <- approxfun(lbound.Ff_c, ys, method = "linear");
lQf_c.func <- approxfun(ubound.Ff_c, ys, method = "linear");

# QTE confidence bands;

taus <- c(3:(nind-2))/(nind+1)
#taus <- c(2:(nind-1))/(nind+1)
#taus <- c(5:(nind-4))/(nind+1)

QTE   <- Qm.func(taus) - Qf.func(taus);
uQTE  <- uQm.func(taus) - lQf.func(taus);
lQTE  <- lQm.func(taus) - uQf.func(taus);

QTE_d   <- Qf_c.func(taus) - Qf.func(taus);
uQTE_d  <- uQf_c.func(taus) - lQf.func(taus);
lQTE_d  <- lQf_c.func(taus) - uQf.func(taus);

QTE_c   <- Qm.func(taus) - Qf_c.func(taus);
uQTE_c  <- uQm.func(taus) - lQf_c.func(taus);
lQTE_c  <- lQm.func(taus) - uQf_c.func(taus);

# Figures of results;

pdf("Gender-gap-dist2.pdf", pointsize=15,  paper = "a4", width=8.0,height=8.0);

par(mfrow=c(1,1));

plot(range(ys), c(0,1), type="n",col=1, xlab="Log of hourly wage", lwd=2,
     ylab="Probability", 
     main="Observed and Counterfactual Distributions (95% CI) ",
     sub=" ");

polygon(c(ys,rev(ys)),c(ubound.Ff,rev(lbound.Ff)), density=100, border=F, 
        col='light blue', lty = 1, lwd = 1);
lines(ys, Ff, lwd=1, col = 4  );



polygon(c(ys,rev(ys)),c(ubound.Ff_c,rev(lbound.Ff_c)), density=100, border=F, 
        col='light grey', lty = 1, lwd = 1);
lines(ys, Ff_c, lwd=1, col = 'dark grey'  );

polygon(c(ys,rev(ys)),c(ubound.Fm,rev(lbound.Fm)), density=100, border=F, 
        col='light green', lty = 1, lwd = 1);
lines(ys, Fm, lwd=1, col = 'dark green'  );



legend(quantile(ys, .01), 1, c(' ', ' ',' ' ), col = c('light blue','light green','light grey'), lwd = c(4,4,4), horiz = F, bty = 'n');
legend(quantile(ys, .01), 1, c('Observed women distribution','Observed men distribution', 'Counterfactual distribution'), col = c(4,'dark green', 'dark grey'), lwd = c(1,1,1), horiz = F, bty = 'n');


dev.off();


pdf("Gender-gap-quant2.pdf", pointsize=15,  paper = "a4", width=8.0,height=8.0);

par(mfrow=c(1,1));

plot(range(taus), range(ys),  type="n",col=1, xlab="Quantile index", lwd=2,
     ylab="Log of hourly wage", 
     main="Observed and Counterfactual Quantiles (95% CI) ",
     sub=" ");

polygon(c(taus,rev(taus)),c(uQf.func(taus),rev(lQf.func(taus))), density=100, border=F, 
        col='light blue', lty = 1, lwd = 1);
lines(taus, Qf.func(taus), lwd=1, col = 4  );

polygon(c(taus,rev(taus)),c(uQf_c.func(taus),rev(lQf_c.func(taus))), density=100, border=F, 
        col='light grey', lty = 1, lwd = 1);
lines(taus, Qf_c.func(taus), lwd=1, col = 'dark grey'  );

polygon(c(taus,rev(taus)),c(uQm.func(taus),rev(lQm.func(taus))), density=100, border=F, 
        col='light green', lty = 1, lwd = 1);
lines(taus, Qm.func(taus), lwd=1, col = 'dark green'  );


legend(min(taus), max(ys), c(' ', ' ',' '), col = c('light blue','light green','light grey'), lwd = c(4,4,4), horiz = F, bty = 'n');
legend(min(taus), max(ys), c('Observed women quantiles', 'Observed men quantiles', 'Counterfactual quantiles'), col = c(4,'dark green','dark grey'), lwd = c(1,1,1), horiz = F, bty = 'n');


dev.off();



pdf("Gender-gap-qte-decomposition.pdf", pointsize=15,  paper = "a4", width=5,height=10.0);

par(mfrow=c(3,1));

plot(range(taus), c(-0.1,0.5),  type="n",col=1, xlab="Quantile index", lwd=2,
     ylab="Difference in log of hourly wage", 
     main="Gender wage gap (90% CI) ",
     sub=" ");

polygon(c(taus,rev(taus)),c(uQTE,rev(lQTE)), density=100, border=F, 
        col='light blue', lty = 1, lwd = 1);
lines(taus, QTE, lwd=1, col = 4  );

legend(min(taus), .5, c(' '), col = c('light blue'), lwd = c(4), horiz = F, bty = 'n');
legend(min(taus), .5, c('QE'), col = c(4), lwd = c(1,1), horiz = F, bty = 'n');


plot(range(taus), c(-0.1,0.5),  type="n",col=1, xlab="Quantile index", lwd=2,
     ylab="Difference in log of hourly wage", 
     main="Discrimination (90% CI) ",
     sub=" ");

polygon(c(taus,rev(taus)),c(uQTE_d,rev(lQTE_d)), density=100, border=F, 
        col='light blue', lty = 1, lwd = 1);
lines(taus, QTE_d, lwd=1, col = 4  );

legend(min(taus), .5, c(' '), col = c('light blue'), lwd = c(4), horiz = F, bty = 'n');
legend(min(taus), .5, c('QE'), col = c(4), lwd = c(1,1), horiz = F, bty = 'n');


plot(range(taus), c(-0.1,0.5),  type="n",col=1, xlab="Quantile index", lwd=2,
     ylab="Difference in log of hourly wage", 
     main="Composition (90% CI) ",
     sub=" ");

polygon(c(taus,rev(taus)),c(uQTE_c,rev(lQTE_c)), density=100, border=F, 
        col='light blue', lty = 1, lwd = 1);
lines(taus, QTE_c, lwd=1, col = 4  );

legend(min(taus), .5, c(' '), col = c('light blue'), lwd = c(4), horiz = F, bty = 'n');
legend(min(taus), .5, c('QE'), col = c(4), lwd = c(1,1), horiz = F, bty = 'n');



dev.off();




pdf("Gender-gap-hists.pdf", pointsize=15,  paper = "a4", width=8.0,height=8.0);

par(mfrow=c(1,2));

hist(lnw[female==1], breaks = unique(lnw[female==1]), col=2, xlab="Log of hourly wage", lwd=2,
     ylab="Frequency", 
     main="Women log hourly wages",
     sub=" ");

hist(lnw[female==0], breaks = unique(lnw[female==0]), col=2, xlab="Log of hourly wage", lwd=2,
     ylab="Frequency", 
     main="Men log hourly wages",
     sub=" ");



dev.off();



