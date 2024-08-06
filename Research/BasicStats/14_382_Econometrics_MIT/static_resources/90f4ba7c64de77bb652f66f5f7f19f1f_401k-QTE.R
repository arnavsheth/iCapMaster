#################################################################################################
#  This is an example of estimation of ATEs  
#  14.382 L10 MIT.  V. Chernozhukov, I. Fernandez-Val
#
# Data source: SIPP 1991 (Abadie, 2003)

# Description of the data: the sample selection and variable contruction follow

# Abadie, Alberto (2003), "Semiparametric instrumental variable estimation of treatment response 
# models," Journal of Econometrics, Elsevier, vol. 113(2), pages 231-263, April.

# The variables in the data set include:

# net_tfa:  net total financial assets
# e401:     = 1 if employer offers 401(k)
# age
# inc:      income
# fsize:    family size
# educ:     years of education
# db:       = 1 if indivuduals has defined benefit pension
# marr:     = 1 if married
# twoearn:  = 1 if two-earner household
# pira:     = 1 if individual participates in IRA plan
# hown      = 1 if home owner
#################################################################################################

# Cross validation uses 1se rule and union of selected variables for all the components;
# Bootstrap weights are recentered to have zero mean;

# Created on May 3, 2015;

# Loading libraries;

library(foreign);
library(quantreg);
library(splines);
library(lattice);
library(matlab);
library(mnormt);
library(glmnet);
library(MASS);
library(xtable);
library(sandwich);
library(quantreg);

# Loading functions;

source("/Users/ivan/Dropbox/shared/14.382/programs/401k/R-NPA-functions-1se-union-rweight.R")

# Reading data set;

data  <- read.dta("/Users/ivan/Dropbox/shared/14.382/data/401k/sipp1991.dta");
attach(data);
n     <- nrow(data); 
B     <- 200;
alpha <- .05;
set.seed(888);

setwd("/Users/ivan/Dropbox/shared/14.382/results/");

#######################################################################################;
# DEPENDENT VARIABLE: NET TOTAL FINANCIAL ASSETS;
#######################################################################################;

# Descriptive statistics;

options(digits=2);
dstats <- cbind(sapply(data, mean), sapply(data, sd), sapply(data[e401==1, ], mean), sapply(data[e401==0, ], mean));
xtable(dstats, caption = "Descriptive Statistics", label = "dstats");

#######################################################################################;
# QUANTILE AND DISTRIBUTION EFFECTS;
#######################################################################################;

# Creating matrix of indicators;

taus  <- c(1:99)/100;
rang <- (taus >= .05) & (taus <= .95);

ys    <- as.matrix(quantile(net_tfa, taus));
nys    <- nrow(ys);
dys		<- 1*(kronecker(ones(1,nys), as.matrix(net_tfa)) <= kronecker(t(ys), ones(n,1)));


#------------------------------------------------------------------------------------#;
# 1 - Specification without interactions;
#------------------------------------------------------------------------------------#;


# Linear quantile regression;

# form_y    <- "net_tfa";
# form_z    <- "e401";
# form_x    <- "I(age < 30) + I((age >= 30) & (age <36)) + I((age >= 36) & (age <45)) + I((age >= 45) & (age <55)) + 
# I(inc < 10000) + I((inc >= 10000) & (inc < 20000)) + I((inc >= 20000) & (inc < 30000)) + I((inc >= 30000) & (inc < 40000)) + I((inc >= 40000) & (inc < 50000)) + I((inc >= 50000) & (inc < 75000)) + 
# I(educ < 12) + I(educ == 12) + I((educ > 12) & (educ < 16)) +
# fsize + marr + twoearn + db + pira + hown";
# 
# lm.fit.y   <- lm(paste(form_y, "~", form_x));
# HCV.coefs  <- vcovHC(lm.fit.y, type = 'HC');
# coefs.se.y <- sqrt(diag(HCV.coefs)); # White std errors
# ry         <- lm.fit.y$res;
# 
# lm.fit.z    <- lm(paste(form_z, "~", form_x));
# HCV.coefs   <- vcovHC(lm.fit.z, type = 'HC');
# coefs.se.z  <- sqrt(diag(HCV.coefs)); # White std errors
# rz          <- lm.fit.z$res;
# 
# lm.fit.ry    <- lm(ry ~ rz - 1);
# ate.ols      <- lm.fit.ry$coef;
# HCV.coefs    <- vcovHC(lm.fit.ry, type = 'HC');
# se.ate.ols   <- sqrt(diag(HCV.coefs)); # White std errors


# Nonparametric estimation;

# Choosing specification;

form_y    <- "data$dy";
form_x    <- "poly(age, 6) + poly(inc, 8) + poly(educ, 4) + poly(fsize,2) + as.factor(marr) + as.factor(twoearn) + as.factor(db) + as.factor(pira) + as.factor(hown)";

lasso     <- FALSE;

# Nonparametric estimation of components;

cond.comp <- COND.COMP.LOGIT(data, dys, form_x, form_y, lasso = lasso);
cond.cdf  <- COND.CDF.LOGIT(data, dys, form_x, form_y, lasso = lasso, cond.comp$selected);

# Estimation of quantile effects;

dsf1   <- NULL;
dsft1  <- NULL;

se.dsf1   <- NULL;
se.dsft1  <- NULL;

dsf0   <- NULL;
dsft0  <- NULL;

se.dsf0   <- NULL;
se.dsft0  <- NULL;

for (i in 1:nys)
{
  data$dy <-  dys[,i]; 
  
  # Estimation of distribution effects;
    
  dsf1      <- c(dsf1, ASF(data$dy, e401, cond.cdf$my_z1x[,i], cond.cdf$my_z0x[,i], cond.comp$mz_x, d0 = 1) );
  dsft1     <- c(dsft1, ASFT(data$dy, e401, cond.cdf$my_z0x[,i], cond.comp$mz_x, cond.comp$mz, d0 = 1) );
  
  se.dsf1      <- c(se.dsf1, SE.ASF(data$dy, e401, cond.cdf$my_z1x[,i], cond.cdf$my_z0x[,i], cond.comp$mz_x, d0 = 1) );
  se.dsft1     <- c(se.dsft1, SE.ASFT(data$dy, e401, cond.cdf$my_z0x[,i], cond.comp$mz_x, cond.comp$mz, d0 = 1) );
  
  dsf0      <- c(dsf0, ASF(data$dy, e401, cond.cdf$my_z1x[,i], cond.cdf$my_z0x[,i], cond.comp$mz_x, d0 = 0) );
  dsft0     <- c(dsft0, ASFT(data$dy, e401, cond.cdf$my_z0x[,i], cond.comp$mz_x, cond.comp$mz, d0 = 0) );
  
  se.dsf0      <- c(se.dsf0, SE.ASF(data$dy, e401, cond.cdf$my_z1x[,i], cond.cdf$my_z0x[,i], cond.comp$mz_x, d0 = 0) );
  se.dsft0     <- c(se.dsft0, SE.ASFT(data$dy, e401, cond.cdf$my_z0x[,i], cond.comp$mz_x, cond.comp$mz, d0 = 0) );
  
}


qsf1    <- approxfun(dsf1, ys, yleft = -Inf, yright = Inf);
qsft1   <- approxfun(dsft1, ys, yleft = -Inf, yright = Inf);

qsf0    <- approxfun(dsf0, ys, yleft = -Inf, yright = Inf);
qsft0   <- approxfun(dsft0, ys, yleft = -Inf, yright = Inf);

qte     <- qsf1(taus)   - qsf0(taus);
qtt     <- qsft1(taus)  - qsft0(taus);

qe.bootstrap <- QE.bootstrap(data=data, n = n, dys = dys, taus = taus, ys = ys,  B=B, cond.comp, cond.cdf, qte, qtt, dsf1, dsf0, dsft1, dsft0, alpha = .05, rang);

# qtes     <- qte;
# qtts     <- qtt;

# lci.qtes  <- qte  - qe.bootstrap$talpha.qte  * qe.bootstrap$se.qte;
# lci.qtts  <- qtt  - qe.bootstrap$talpha.qtt  * qe.bootstrap$se.qtt;
# 
# uci.qtes  <- qte  + qe.bootstrap$talpha.qte  * qe.bootstrap$se.qte;
# uci.qtts  <- qtt  + qe.bootstrap$talpha.qtt  * qe.bootstrap$se.qtt;

lci.dsf1  <- dsf1  - qe.bootstrap$talpha.dsf1  * qe.bootstrap$se.dsf1;
lci.dsf0  <- dsf0  - qe.bootstrap$talpha.dsf0  * qe.bootstrap$se.dsf0;

uci.dsf1  <- dsf1  + qe.bootstrap$talpha.dsf1  * qe.bootstrap$se.dsf1;
uci.dsf0  <- dsf0  + qe.bootstrap$talpha.dsf0  * qe.bootstrap$se.dsf0;

lci.dsft1  <- dsft1  - qe.bootstrap$talpha.dsft1  * qe.bootstrap$se.dsft1;
lci.dsft0  <- dsft0  - qe.bootstrap$talpha.dsft0  * qe.bootstrap$se.dsft0;

uci.dsft1  <- dsft1  + qe.bootstrap$talpha.dsft1  * qe.bootstrap$se.dsft1;
uci.dsft0  <- dsft0  + qe.bootstrap$talpha.dsft0  * qe.bootstrap$se.dsft0;

# Create step functions for distributions;

dsf1.func  <- cdf(ys, dsf1);
udsf1.func <- cdf(ys, uci.dsf1);
ldsf1.func <- cdf(ys, lci.dsf1);

dsf0.func  <- cdf(ys, dsf0);
udsf0.func <- cdf(ys, uci.dsf0);
ldsf0.func <- cdf(ys, lci.dsf0);

dsft1.func  <- cdf(ys, dsft1);
udsft1.func <- cdf(ys, uci.dsft1);
ldsft1.func <- cdf(ys, lci.dsft1);

dsft0.func  <- cdf(ys, dsft0);
udsft0.func <- cdf(ys, uci.dsft0);
ldsft0.func <- cdf(ys, lci.dsft0);


# Quantile functions;

qsf1.func   <- left.inv(ys, dsf1, rule=2:1);
lqsf1.func  <- left.inv(ys, uci.dsf1, rule=2:2);
uqsf1.func  <- left.inv(ys, lci.dsf1, rule=2:2);

qsf0.func   <- left.inv(ys, dsf0, rule=2:1);
lqsf0.func  <- left.inv(ys, uci.dsf0, rule=2:2);
uqsf0.func  <- left.inv(ys, lci.dsf0, rule=2:2);

qsft1.func   <- left.inv(ys, dsft1, rule=2:1);
lqsft1.func  <- left.inv(ys, uci.dsft1, rule=2:2);
uqsft1.func  <- left.inv(ys, lci.dsft1, rule=2:2);

qsft0.func   <- left.inv(ys, dsft0, rule=2:1);
lqsft0.func  <- left.inv(ys, uci.dsft0, rule=2:2);
uqsft0.func  <- left.inv(ys, lci.dsft0, rule=2:2);

# Figures of results;

pdf("QTE-nointer-noselec.pdf", pointsize=15,  paper = "a4", width=8.0,height=8.0);

par(mfrow=c(1,1));

curve(qsf1.func(x)-qsf0.func(x), 0.05, .95, type="s", col="dark blue", xlab="Quantile index", ylab="Difference in net_tfa", 
      ylim= c(0,65000), main="QTE (90% CI) ",
      sub=" ");
curve(uqsf1.func(x) - lqsf0.func(x), 0.05, .95, type="s", col="light blue", lty = 1, add=TRUE);
curve(lqsf1.func(x) - uqsf0.func(x), 0.05, .95, type="s", col="light blue", lty = 1, add=TRUE);

abline(h=0);

legend(0.05, 65000, c(' '), col = c('light blue'), lwd = c(4), horiz = F, bty = 'n');
legend(0.05, 65000, c('QE'), col = c(4), lwd = c(1,1), horiz = F, bty = 'n');


dev.off();


pdf("QTT-nointer-noselec.pdf", pointsize=15,  paper = "a4", width=8.0,height=8.0);

par(mfrow=c(1,1));

curve(qsft1.func(x)-qsft0.func(x), 0.05, .95, type="s", col="dark blue", xlab="Quantile index", ylab="Difference in net_tfa", 
      ylim= c(0,65000), main="QTT (90% CI) ",
      sub=" ");
curve(uqsft1.func(x) - lqsft0.func(x), 0.05, .95, type="s", col="light blue", lty = 1, add=TRUE);
curve(lqsft1.func(x) - uqsft0.func(x), 0.05, .95, type="s", col="light blue", lty = 1, add=TRUE);

abline(h=0);

legend(0.05, 65000, c(' '), col = c('light blue'), lwd = c(4), horiz = F, bty = 'n');
legend(0.05, 65000, c('QE'), col = c(4), lwd = c(1,1), horiz = F, bty = 'n');


dev.off();






# Nonparametric estimation with lasso selection;

# Choosing specification;

form_y    <- "data$dy";
form_x    <- "poly(age, 6) + poly(inc, 8) + poly(educ, 4) + poly(fsize,2) + as.factor(marr) + as.factor(twoearn) + as.factor(db) + as.factor(pira) + as.factor(hown)";
lasso     <- TRUE;

# Nonparametric estimation of components;

cond.comp <- COND.COMP.LOGIT(data, dys, form_x, form_y, lasso = lasso);
cond.cdf  <- COND.CDF.LOGIT(data, dys, form_x, form_y, lasso = lasso, cond.comp$selected);

# Estimation of quantile effects;

dsf1   <- NULL;
dsft1  <- NULL;

se.dsf1   <- NULL;
se.dsft1  <- NULL;

dsf0   <- NULL;
dsft0  <- NULL;

se.dsf0   <- NULL;
se.dsft0  <- NULL;

for (i in 1:nys)
{
  data$dy <-  dys[,i]; 
  
  # Estimation of distribution effects;
  
  dsf1      <- c(dsf1, ASF(data$dy, e401, cond.cdf$my_z1x[,i], cond.cdf$my_z0x[,i], cond.comp$mz_x, d0 = 1) );
  dsft1     <- c(dsft1, ASFT(data$dy, e401, cond.cdf$my_z0x[,i], cond.comp$mz_x, cond.comp$mz, d0 = 1) );
  
  se.dsf1      <- c(se.dsf1, SE.ASF(data$dy, e401, cond.cdf$my_z1x[,i], cond.cdf$my_z0x[,i], cond.comp$mz_x, d0 = 1) );
  se.dsft1     <- c(se.dsft1, SE.ASFT(data$dy, e401, cond.cdf$my_z0x[,i], cond.comp$mz_x, cond.comp$mz, d0 = 1) );
  
  dsf0      <- c(dsf0, ASF(data$dy, e401, cond.cdf$my_z1x[,i], cond.cdf$my_z0x[,i], cond.comp$mz_x, d0 = 0) );
  dsft0     <- c(dsft0, ASFT(data$dy, e401, cond.cdf$my_z0x[,i], cond.comp$mz_x, cond.comp$mz, d0 = 0) );
  
  se.dsf0      <- c(se.dsf0, SE.ASF(data$dy, e401, cond.cdf$my_z1x[,i], cond.cdf$my_z0x[,i], cond.comp$mz_x, d0 = 0) );
  se.dsft0     <- c(se.dsft0, SE.ASFT(data$dy, e401, cond.cdf$my_z0x[,i], cond.comp$mz_x, cond.comp$mz, d0 = 0) );
  
}


qsf1    <- approxfun(dsf1, ys, yleft = -Inf, yright = Inf);
qsft1   <- approxfun(dsft1, ys, yleft = -Inf, yright = Inf);

qsf0    <- approxfun(dsf0, ys, yleft = -Inf, yright = Inf);
qsft0   <- approxfun(dsft0, ys, yleft = -Inf, yright = Inf);

qte     <- qsf1(taus)   - qsf0(taus);
qtt     <- qsft1(taus)  - qsft0(taus);

qe.bootstrap <- QE.bootstrap(data=data, n = n, dys = dys, taus = taus, ys = ys,  B=B, cond.comp, cond.cdf, qte, qtt, dsf1, dsf0, dsft1, dsft0, alpha = .05, rang);

# qtes     <- qte;
# qtts     <- qtt;

# lci.qtes  <- qte  - qe.bootstrap$talpha.qte  * qe.bootstrap$se.qte;
# lci.qtts  <- qtt  - qe.bootstrap$talpha.qtt  * qe.bootstrap$se.qtt;
# 
# uci.qtes  <- qte  + qe.bootstrap$talpha.qte  * qe.bootstrap$se.qte;
# uci.qtts  <- qtt  + qe.bootstrap$talpha.qtt  * qe.bootstrap$se.qtt;

lci.dsf1  <- dsf1  - qe.bootstrap$talpha.dsf1  * qe.bootstrap$se.dsf1;
lci.dsf0  <- dsf0  - qe.bootstrap$talpha.dsf0  * qe.bootstrap$se.dsf0;

uci.dsf1  <- dsf1  + qe.bootstrap$talpha.dsf1  * qe.bootstrap$se.dsf1;
uci.dsf0  <- dsf0  + qe.bootstrap$talpha.dsf0  * qe.bootstrap$se.dsf0;

lci.dsft1  <- dsft1  - qe.bootstrap$talpha.dsft1  * qe.bootstrap$se.dsft1;
lci.dsft0  <- dsft0  - qe.bootstrap$talpha.dsft0  * qe.bootstrap$se.dsft0;

uci.dsft1  <- dsft1  + qe.bootstrap$talpha.dsft1  * qe.bootstrap$se.dsft1;
uci.dsft0  <- dsft0  + qe.bootstrap$talpha.dsft0  * qe.bootstrap$se.dsft0;

# Create step functions for distributions;

dsf1.func  <- cdf(ys, dsf1);
udsf1.func <- cdf(ys, uci.dsf1);
ldsf1.func <- cdf(ys, lci.dsf1);

dsf0.func  <- cdf(ys, dsf0);
udsf0.func <- cdf(ys, uci.dsf0);
ldsf0.func <- cdf(ys, lci.dsf0);

dsft1.func  <- cdf(ys, dsft1);
udsft1.func <- cdf(ys, uci.dsft1);
ldsft1.func <- cdf(ys, lci.dsft1);

dsft0.func  <- cdf(ys, dsft0);
udsft0.func <- cdf(ys, uci.dsft0);
ldsft0.func <- cdf(ys, lci.dsft0);


# Quantile functions;

qsf1.func   <- left.inv(ys, dsf1, rule=2:1);
lqsf1.func  <- left.inv(ys, uci.dsf1, rule=2:2);
uqsf1.func  <- left.inv(ys, lci.dsf1, rule=2:2);

qsf0.func   <- left.inv(ys, dsf0, rule=2:1);
lqsf0.func  <- left.inv(ys, uci.dsf0, rule=2:2);
uqsf0.func  <- left.inv(ys, lci.dsf0, rule=2:2);

qsft1.func   <- left.inv(ys, dsft1, rule=2:1);
lqsft1.func  <- left.inv(ys, uci.dsft1, rule=2:2);
uqsft1.func  <- left.inv(ys, lci.dsft1, rule=2:2);

qsft0.func   <- left.inv(ys, dsft0, rule=2:1);
lqsft0.func  <- left.inv(ys, uci.dsft0, rule=2:2);
uqsft0.func  <- left.inv(ys, lci.dsft0, rule=2:2);

# Figures of results;

pdf("QTE-nointer-selec.pdf", pointsize=15,  paper = "a4", width=8.0,height=8.0);

par(mfrow=c(1,1));

curve(qsf1.func(x)-qsf0.func(x), 0.05, .95, type="s", col="dark blue", xlab="Quantile index", ylab="Difference in net_tfa", 
      ylim= c(0,65000), main="QTE (90% CI) ",
      sub=" ");
curve(uqsf1.func(x) - lqsf0.func(x), 0.05, .95, type="s", col="light blue", lty = 1, add=TRUE);
curve(lqsf1.func(x) - uqsf0.func(x), 0.05, .95, type="s", col="light blue", lty = 1, add=TRUE);

abline(h=0);

legend(0.05, 65000, c(' '), col = c('light blue'), lwd = c(4), horiz = F, bty = 'n');
legend(0.05, 65000, c('QE'), col = c(4), lwd = c(1,1), horiz = F, bty = 'n');


dev.off();


pdf("QTT-nointer-selec.pdf", pointsize=15,  paper = "a4", width=8.0,height=8.0);

par(mfrow=c(1,1));

curve(qsft1.func(x)-qsft0.func(x), 0.05, .95, type="s", col="dark blue", xlab="Quantile index", ylab="Difference in net_tfa", 
      ylim= c(0,65000), main="QTT (90% CI) ",
      sub=" ");
curve(uqsft1.func(x) - lqsft0.func(x), 0.05, .95, type="s", col="light blue", lty = 1, add=TRUE);
curve(lqsft1.func(x) - uqsft0.func(x), 0.05, .95, type="s", col="light blue", lty = 1, add=TRUE);

abline(h=0);

legend(0.05, 65000, c(' '), col = c('light blue'), lwd = c(4), horiz = F, bty = 'n');
legend(0.05, 65000, c('QE'), col = c(4), lwd = c(1,1), horiz = F, bty = 'n');


dev.off();








#------------------------------------------------------------------------------------#;
# 2 - Specification with two-way interactions;
#------------------------------------------------------------------------------------#;


# # Linear regression;
# 
# form_y    <- "net_tfa";
# form_z    <- "e401";
# form_x    <- "(I(age < 30) + I((age >= 30) & (age <36)) + I((age >= 36) & (age <45)) + I((age >= 45) & (age <55)) + 
# I(inc < 10000) + I((inc >= 10000) & (inc < 20000)) + I((inc >= 20000) & (inc < 30000)) + I((inc >= 30000) & (inc < 40000)) + I((inc >= 40000) & (inc < 50000)) + I((inc >= 50000) & (inc < 75000)) + 
# I(educ < 12) + I(educ == 12) + I((educ > 12) & (educ < 16)) +
# fsize + marr + twoearn + db + pira + hown)^2";
# 
# 
# lm.fit.y   <- lm(paste(form_y, "~", form_x));
# #HCV.coefs  <- vcovHC(lm.fit.y, type = 'HC');
# #coefs.se.y <- sqrt(diag(HCV.coefs)); # White std errors
# ry         <- lm.fit.y$res;
# 
# lm.fit.z    <- lm(paste(form_z, "~", form_x));
# #HCV.coefs   <- vcovHC(lm.fit.z, type = 'HC');
# #coefs.se.z  <- sqrt(diag(HCV.coefs)); # White std errors
# rz          <- lm.fit.z$res;
# 
# lm.fit.ry    <- lm(ry ~ rz - 1);
# ate.ols      <- lm.fit.ry$coef;
# HCV.coefs    <- vcovHC(lm.fit.ry, type = 'HC');
# se.ate.ols   <- sqrt(diag(HCV.coefs)); # White std errors


# Nonparametric estimation;

form_y    <- "data$dy";
form_x    <- "(poly(age, 6) + poly(inc, 8) + poly(educ, 4) + poly(fsize,2) + as.factor(marr) + as.factor(twoearn) + as.factor(db) + as.factor(pira) + as.factor(hown))^2";
lasso     <- FALSE;

# Nonparametric estimation of components;

cond.comp <- COND.COMP.LOGIT(data, dys, form_x, form_y, lasso = lasso);
cond.cdf  <- COND.CDF.LOGIT(data, dys, form_x, form_y, lasso = lasso, cond.comp$selected);

# Estimation of quantile effects;

dsf1   <- NULL;
dsft1  <- NULL;

se.dsf1   <- NULL;
se.dsft1  <- NULL;

dsf0   <- NULL;
dsft0  <- NULL;

se.dsf0   <- NULL;
se.dsft0  <- NULL;

for (i in 1:nys)
{
  data$dy <-  dys[,i]; 
  
  # Estimation of distribution effects;
  
  dsf1      <- c(dsf1, ASF(data$dy, e401, cond.cdf$my_z1x[,i], cond.cdf$my_z0x[,i], cond.comp$mz_x, d0 = 1) );
  dsft1     <- c(dsft1, ASFT(data$dy, e401, cond.cdf$my_z0x[,i], cond.comp$mz_x, cond.comp$mz, d0 = 1) );
  
  se.dsf1      <- c(se.dsf1, SE.ASF(data$dy, e401, cond.cdf$my_z1x[,i], cond.cdf$my_z0x[,i], cond.comp$mz_x, d0 = 1) );
  se.dsft1     <- c(se.dsft1, SE.ASFT(data$dy, e401, cond.cdf$my_z0x[,i], cond.comp$mz_x, cond.comp$mz, d0 = 1) );
  
  dsf0      <- c(dsf0, ASF(data$dy, e401, cond.cdf$my_z1x[,i], cond.cdf$my_z0x[,i], cond.comp$mz_x, d0 = 0) );
  dsft0     <- c(dsft0, ASFT(data$dy, e401, cond.cdf$my_z0x[,i], cond.comp$mz_x, cond.comp$mz, d0 = 0) );
  
  se.dsf0      <- c(se.dsf0, SE.ASF(data$dy, e401, cond.cdf$my_z1x[,i], cond.cdf$my_z0x[,i], cond.comp$mz_x, d0 = 0) );
  se.dsft0     <- c(se.dsft0, SE.ASFT(data$dy, e401, cond.cdf$my_z0x[,i], cond.comp$mz_x, cond.comp$mz, d0 = 0) );
  
}


qsf1    <- approxfun(dsf1, ys, yleft = -Inf, yright = Inf);
qsft1   <- approxfun(dsft1, ys, yleft = -Inf, yright = Inf);

qsf0    <- approxfun(dsf0, ys, yleft = -Inf, yright = Inf);
qsft0   <- approxfun(dsft0, ys, yleft = -Inf, yright = Inf);

qte     <- qsf1(taus)   - qsf0(taus);
qtt     <- qsft1(taus)  - qsft0(taus);

qe.bootstrap <- QE.bootstrap(data=data, n = n, dys = dys, taus = taus, ys = ys,  B=B, cond.comp, cond.cdf, qte, qtt, dsf1, dsf0, dsft1, dsft0, alpha = .05, rang);

# qtes     <- qte;
# qtts     <- qtt;

# lci.qtes  <- qte  - qe.bootstrap$talpha.qte  * qe.bootstrap$se.qte;
# lci.qtts  <- qtt  - qe.bootstrap$talpha.qtt  * qe.bootstrap$se.qtt;
# 
# uci.qtes  <- qte  + qe.bootstrap$talpha.qte  * qe.bootstrap$se.qte;
# uci.qtts  <- qtt  + qe.bootstrap$talpha.qtt  * qe.bootstrap$se.qtt;

lci.dsf1  <- dsf1  - qe.bootstrap$talpha.dsf1  * qe.bootstrap$se.dsf1;
lci.dsf0  <- dsf0  - qe.bootstrap$talpha.dsf0  * qe.bootstrap$se.dsf0;

uci.dsf1  <- dsf1  + qe.bootstrap$talpha.dsf1  * qe.bootstrap$se.dsf1;
uci.dsf0  <- dsf0  + qe.bootstrap$talpha.dsf0  * qe.bootstrap$se.dsf0;

lci.dsft1  <- dsft1  - qe.bootstrap$talpha.dsft1  * qe.bootstrap$se.dsft1;
lci.dsft0  <- dsft0  - qe.bootstrap$talpha.dsft0  * qe.bootstrap$se.dsft0;

uci.dsft1  <- dsft1  + qe.bootstrap$talpha.dsft1  * qe.bootstrap$se.dsft1;
uci.dsft0  <- dsft0  + qe.bootstrap$talpha.dsft0  * qe.bootstrap$se.dsft0;

# Create step functions for distributions;

dsf1.func  <- cdf(ys, dsf1);
udsf1.func <- cdf(ys, uci.dsf1);
ldsf1.func <- cdf(ys, lci.dsf1);

dsf0.func  <- cdf(ys, dsf0);
udsf0.func <- cdf(ys, uci.dsf0);
ldsf0.func <- cdf(ys, lci.dsf0);

dsft1.func  <- cdf(ys, dsft1);
udsft1.func <- cdf(ys, uci.dsft1);
ldsft1.func <- cdf(ys, lci.dsft1);

dsft0.func  <- cdf(ys, dsft0);
udsft0.func <- cdf(ys, uci.dsft0);
ldsft0.func <- cdf(ys, lci.dsft0);


# Quantile functions;

qsf1.func   <- left.inv(ys, dsf1, rule=2:1);
lqsf1.func  <- left.inv(ys, uci.dsf1, rule=2:2);
uqsf1.func  <- left.inv(ys, lci.dsf1, rule=2:2);

qsf0.func   <- left.inv(ys, dsf0, rule=2:1);
lqsf0.func  <- left.inv(ys, uci.dsf0, rule=2:2);
uqsf0.func  <- left.inv(ys, lci.dsf0, rule=2:2);

qsft1.func   <- left.inv(ys, dsft1, rule=2:1);
lqsft1.func  <- left.inv(ys, uci.dsft1, rule=2:2);
uqsft1.func  <- left.inv(ys, lci.dsft1, rule=2:2);

qsft0.func   <- left.inv(ys, dsft0, rule=2:1);
lqsft0.func  <- left.inv(ys, uci.dsft0, rule=2:2);
uqsft0.func  <- left.inv(ys, lci.dsft0, rule=2:2);

# Figures of results;

pdf("QTE-inter-noselec.pdf", pointsize=15,  paper = "a4", width=8.0,height=8.0);

par(mfrow=c(1,1));

curve(qsf1.func(x)-qsf0.func(x), 0.05, .95, type="s", col="dark blue", xlab="Quantile index", ylab="Difference in net_tfa", 
      ylim= c(0,65000), main="QTE (90% CI) ",
      sub=" ");
curve(uqsf1.func(x) - lqsf0.func(x), 0.05, .95, type="s", col="light blue", lty = 1, add=TRUE);
curve(lqsf1.func(x) - uqsf0.func(x), 0.05, .95, type="s", col="light blue", lty = 1, add=TRUE);

abline(h=0);

legend(0.05, 65000, c(' '), col = c('light blue'), lwd = c(4), horiz = F, bty = 'n');
legend(0.05, 65000, c('QE'), col = c(4), lwd = c(1,1), horiz = F, bty = 'n');


dev.off();


pdf("QTT-inter-noselec.pdf", pointsize=15,  paper = "a4", width=8.0,height=8.0);

par(mfrow=c(1,1));

curve(qsft1.func(x)-qsft0.func(x), 0.05, .95, type="s", col="dark blue", xlab="Quantile index", ylab="Difference in net_tfa", 
      ylim= c(0,65000), main="QTT (90% CI) ",
      sub=" ");
curve(uqsft1.func(x) - lqsft0.func(x), 0.05, .95, type="s", col="light blue", lty = 1, add=TRUE);
curve(lqsft1.func(x) - uqsft0.func(x), 0.05, .95, type="s", col="light blue", lty = 1, add=TRUE);

abline(h=0);

legend(0.05, 65000, c(' '), col = c('light blue'), lwd = c(4), horiz = F, bty = 'n');
legend(0.05, 65000, c('QE'), col = c(4), lwd = c(1,1), horiz = F, bty = 'n');


dev.off();








# Nonparametric estimation with lasso selection;

form_y    <- "data$dy";
form_x    <- "(poly(age, 6) + poly(inc, 8) + poly(educ, 4) + poly(fsize,2) + as.factor(marr) + as.factor(twoearn) + as.factor(db) + as.factor(pira) + as.factor(hown))^2";
lasso     <- TRUE;

# Nonparametric estimation of components;

cond.comp <- COND.COMP.LOGIT(data, dys, form_x, form_y, lasso = lasso);
cond.cdf  <- COND.CDF.LOGIT(data, dys, form_x, form_y, lasso = lasso, cond.comp$selected);

# Estimation of quantile effects;

dsf1   <- NULL;
dsft1  <- NULL;

se.dsf1   <- NULL;
se.dsft1  <- NULL;

dsf0   <- NULL;
dsft0  <- NULL;

se.dsf0   <- NULL;
se.dsft0  <- NULL;

for (i in 1:nys)
{
  data$dy <-  dys[,i]; 
  
  # Estimation of distribution effects;
  
  dsf1      <- c(dsf1, ASF(data$dy, e401, cond.cdf$my_z1x[,i], cond.cdf$my_z0x[,i], cond.comp$mz_x, d0 = 1) );
  dsft1     <- c(dsft1, ASFT(data$dy, e401, cond.cdf$my_z0x[,i], cond.comp$mz_x, cond.comp$mz, d0 = 1) );
  
  se.dsf1      <- c(se.dsf1, SE.ASF(data$dy, e401, cond.cdf$my_z1x[,i], cond.cdf$my_z0x[,i], cond.comp$mz_x, d0 = 1) );
  se.dsft1     <- c(se.dsft1, SE.ASFT(data$dy, e401, cond.cdf$my_z0x[,i], cond.comp$mz_x, cond.comp$mz, d0 = 1) );
  
  dsf0      <- c(dsf0, ASF(data$dy, e401, cond.cdf$my_z1x[,i], cond.cdf$my_z0x[,i], cond.comp$mz_x, d0 = 0) );
  dsft0     <- c(dsft0, ASFT(data$dy, e401, cond.cdf$my_z0x[,i], cond.comp$mz_x, cond.comp$mz, d0 = 0) );
  
  se.dsf0      <- c(se.dsf0, SE.ASF(data$dy, e401, cond.cdf$my_z1x[,i], cond.cdf$my_z0x[,i], cond.comp$mz_x, d0 = 0) );
  se.dsft0     <- c(se.dsft0, SE.ASFT(data$dy, e401, cond.cdf$my_z0x[,i], cond.comp$mz_x, cond.comp$mz, d0 = 0) );
  
}


qsf1    <- approxfun(dsf1, ys, yleft = -Inf, yright = Inf);
qsft1   <- approxfun(dsft1, ys, yleft = -Inf, yright = Inf);

qsf0    <- approxfun(dsf0, ys, yleft = -Inf, yright = Inf);
qsft0   <- approxfun(dsft0, ys, yleft = -Inf, yright = Inf);

qte     <- qsf1(taus)   - qsf0(taus);
qtt     <- qsft1(taus)  - qsft0(taus);

qe.bootstrap <- QE.bootstrap(data=data, n = n, dys = dys, taus = taus, ys = ys,  B=B, cond.comp, cond.cdf, qte, qtt, dsf1, dsf0, dsft1, dsft0, alpha = .05, rang);

# qtes     <- qte;
# qtts     <- qtt;

# lci.qtes  <- qte  - qe.bootstrap$talpha.qte  * qe.bootstrap$se.qte;
# lci.qtts  <- qtt  - qe.bootstrap$talpha.qtt  * qe.bootstrap$se.qtt;
# 
# uci.qtes  <- qte  + qe.bootstrap$talpha.qte  * qe.bootstrap$se.qte;
# uci.qtts  <- qtt  + qe.bootstrap$talpha.qtt  * qe.bootstrap$se.qtt;

lci.dsf1  <- dsf1  - qe.bootstrap$talpha.dsf1  * qe.bootstrap$se.dsf1;
lci.dsf0  <- dsf0  - qe.bootstrap$talpha.dsf0  * qe.bootstrap$se.dsf0;

uci.dsf1  <- dsf1  + qe.bootstrap$talpha.dsf1  * qe.bootstrap$se.dsf1;
uci.dsf0  <- dsf0  + qe.bootstrap$talpha.dsf0  * qe.bootstrap$se.dsf0;

lci.dsft1  <- dsft1  - qe.bootstrap$talpha.dsft1  * qe.bootstrap$se.dsft1;
lci.dsft0  <- dsft0  - qe.bootstrap$talpha.dsft0  * qe.bootstrap$se.dsft0;

uci.dsft1  <- dsft1  + qe.bootstrap$talpha.dsft1  * qe.bootstrap$se.dsft1;
uci.dsft0  <- dsft0  + qe.bootstrap$talpha.dsft0  * qe.bootstrap$se.dsft0;

# Create step functions for distributions;

dsf1.func  <- cdf(ys, dsf1);
udsf1.func <- cdf(ys, uci.dsf1);
ldsf1.func <- cdf(ys, lci.dsf1);

dsf0.func  <- cdf(ys, dsf0);
udsf0.func <- cdf(ys, uci.dsf0);
ldsf0.func <- cdf(ys, lci.dsf0);

dsft1.func  <- cdf(ys, dsft1);
udsft1.func <- cdf(ys, uci.dsft1);
ldsft1.func <- cdf(ys, lci.dsft1);

dsft0.func  <- cdf(ys, dsft0);
udsft0.func <- cdf(ys, uci.dsft0);
ldsft0.func <- cdf(ys, lci.dsft0);


# Quantile functions;

qsf1.func   <- left.inv(ys, dsf1, rule=2:1);
lqsf1.func  <- left.inv(ys, uci.dsf1, rule=2:2);
uqsf1.func  <- left.inv(ys, lci.dsf1, rule=2:2);

qsf0.func   <- left.inv(ys, dsf0, rule=2:1);
lqsf0.func  <- left.inv(ys, uci.dsf0, rule=2:2);
uqsf0.func  <- left.inv(ys, lci.dsf0, rule=2:2);

qsft1.func   <- left.inv(ys, dsft1, rule=2:1);
lqsft1.func  <- left.inv(ys, uci.dsft1, rule=2:2);
uqsft1.func  <- left.inv(ys, lci.dsft1, rule=2:2);

qsft0.func   <- left.inv(ys, dsft0, rule=2:1);
lqsft0.func  <- left.inv(ys, uci.dsft0, rule=2:2);
uqsft0.func  <- left.inv(ys, lci.dsft0, rule=2:2);

# Figures of results;

pdf("QTE-inter-selec.pdf", pointsize=15,  paper = "a4", width=8.0,height=8.0);

par(mfrow=c(1,1));

curve(qsf1.func(x)-qsf0.func(x), 0.05, .95, type="s", col="dark blue", xlab="Quantile index", ylab="Difference in net_tfa", 
      ylim= c(0,65000), main="QTE (90% CI) ",
      sub=" ");
curve(uqsf1.func(x) - lqsf0.func(x), 0.05, .95, type="s", col="light blue", lty = 1, add=TRUE);
curve(lqsf1.func(x) - uqsf0.func(x), 0.05, .95, type="s", col="light blue", lty = 1, add=TRUE);

abline(h=0);

legend(0.05, 65000, c(' '), col = c('light blue'), lwd = c(4), horiz = F, bty = 'n');
legend(0.05, 65000, c('QE'), col = c(4), lwd = c(1,1), horiz = F, bty = 'n');


dev.off();


pdf("QTT-inter-selec.pdf", pointsize=15,  paper = "a4", width=8.0,height=8.0);

par(mfrow=c(1,1));

curve(qsft1.func(x)-qsft0.func(x), 0.05, .95, type="s", col="dark blue", xlab="Quantile index", ylab="Difference in net_tfa", 
      ylim= c(0,65000), main="QTT (90% CI) ",
      sub=" ");
curve(uqsft1.func(x) - lqsft0.func(x), 0.05, .95, type="s", col="light blue", lty = 1, add=TRUE);
curve(lqsft1.func(x) - uqsft0.func(x), 0.05, .95, type="s", col="light blue", lty = 1, add=TRUE);


legend(0.05, 65000, c(' '), col = c('light blue'), lwd = c(4), horiz = F, bty = 'n');
legend(0.05, 65000, c('QE'), col = c(4), lwd = c(1,1), horiz = F, bty = 'n');

abline(h=0);


dev.off();





