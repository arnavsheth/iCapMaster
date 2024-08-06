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

# Created on Apr-28, 2015;

# Loading libraries;

library(foreign);
library(quantreg);
library(matlab);
library(mnormt);
library(glmnet);
library(MASS);
library(xtable);
library(sandwich);


# Loading functions;

source("/Users/ivan/Dropbox/shared/14.382/programs/401k/R-NPA-functions-1se-union-rweight.R")

# Reading data set;

data  <- read.dta("/Users/ivan/Dropbox/shared/14.382/data/401k/sipp1991.dta");
attach(data);
n     <- nrow(data); 
B     <- 500;
alpha <- .05;
set.seed(888);

#######################################################################################;
# DEPENDENT VARIABLE: NET TOTAL FINANCIAL ASSETS;
#######################################################################################;

# Descriptive statistics;

options(digits=2);
dstats <- cbind(sapply(data, mean), sapply(data, sd), sapply(data[e401==1, ], mean), sapply(data[e401==0, ], mean));
xtable(dstats, caption = "Descriptive Statistics", label = "dstats");

#######################################################################################;
# AVERAGE EFFECTS;
#######################################################################################;

#------------------------------------------------------------------------------------#;
# 1 - Specification without interactions;
#------------------------------------------------------------------------------------#;


# Linear regression;

form_y    <- "net_tfa";
form_z    <- "e401";
form_x    <- "poly(age, 6) + poly(inc, 8) + poly(educ, 4) + poly(fsize,2) + as.factor(marr) + as.factor(twoearn) + as.factor(db) + as.factor(pira) + as.factor(hown)";


lm.fit.y   <- lm(paste(form_y, "~", form_x));
HCV.coefs  <- vcovHC(lm.fit.y, type = 'HC');
coefs.se.y <- sqrt(diag(HCV.coefs)); # White std errors
ry         <- lm.fit.y$res;

lm.fit.z    <- lm(paste(form_z, "~", form_x));
HCV.coefs   <- vcovHC(lm.fit.z, type = 'HC');
coefs.se.z  <- sqrt(diag(HCV.coefs)); # White std errors
rz          <- lm.fit.z$res;

lm.fit.ry    <- lm(ry ~ rz - 1);
ate.ols      <- lm.fit.ry$coef;
HCV.coefs    <- vcovHC(lm.fit.ry, type = 'HC');
se.ate.ols   <- sqrt(diag(HCV.coefs)); # White std errors

# Linear regression with lasso selection;

form_y    <- "net_tfa";
form_z    <- "e401";
form_x    <- "poly(age, 6) + poly(inc, 8) + poly(educ, 4) + poly(fsize,2) + as.factor(marr) + as.factor(twoearn) + as.factor(db) + as.factor(pira) + as.factor(hown)";


lm.fit.y        <- lm(paste(form_y, "~", form_x), x = TRUE, y = TRUE);
cv.lasso        <- cv.glmnet(lm.fit.y$x[ ,-1], lm.fit.y$y, family = "gaussian", nfolds = 500);
lasso.fit.y     <- glmnet(lm.fit.y$x[ ,-1], lm.fit.y$y, family = "gaussian", lambda = cv.lasso$lambda.1se);
selected        <- c(TRUE, !as.vector(lasso.fit.y$beta == 0));
plasso.fit.y    <- lm(lm.fit.y$y ~ lm.fit.y$x[ , selected]);
ry              <- plasso.fit.y$res;

lm.fit.z        <- lm(paste(form_z, "~", form_x), x = TRUE, y = TRUE);
cv.lasso        <- cv.glmnet(lm.fit.z$x[ ,-1], lm.fit.z$y, family = "gaussian", nfolds = 500);
lasso.fit.z     <- glmnet(lm.fit.z$x[ ,-1], lm.fit.z$y, family = "gaussian", lambda = cv.lasso$lambda.1se);
selected        <- c(TRUE, !as.vector(lasso.fit.z$beta == 0));
plasso.fit.z    <- lm(lm.fit.z$y ~ lm.fit.z$x[ , selected]);
rz              <- plasso.fit.z$res;


lm.fit.ry      <- lm(ry ~ rz - 1);
ate.ols.s      <- lm.fit.ry$coef;
HCV.coefs      <- vcovHC(lm.fit.ry, type = 'HC');
se.ate.ols.s   <- sqrt(diag(HCV.coefs)); # White std errors


# Nonparametric estimation;

form_y    <- "net_tfa";
form_x    <- "poly(age, 6) + poly(inc, 8) + poly(educ, 4) + poly(fsize,2) + as.factor(marr) + as.factor(twoearn) + as.factor(db) + as.factor(pira) + as.factor(hown)";


lasso     <- FALSE;
cond.comp <- COND.COMP(data, form_x, form_y, lasso = lasso);

ate.logit   <- ATE(net_tfa, e401, cond.comp$my_z1x, cond.comp$my_z0x, cond.comp$mz_x.logit);
att.logit   <- ATT(net_tfa, e401, cond.comp$my_z0x, cond.comp$mz_x.logit, cond.comp$mz);

se.ate.logit   <- SE.ATE(net_tfa, e401, cond.comp$my_z1x, cond.comp$my_z0x, cond.comp$mz_x.logit);
se.att.logit   <- SE.ATT(net_tfa, e401, cond.comp$my_z0x, cond.comp$mz_x.logit, cond.comp$mz);

# Nonparametric estimation with lasso selection;

form_y    <- "net_tfa";
form_x    <- "poly(age, 6) + poly(inc, 8) + poly(educ, 4) + poly(fsize,2) + as.factor(marr) + as.factor(twoearn) + as.factor(db) + as.factor(pira) + as.factor(hown)";


lasso     <- TRUE;
cond.comp <- COND.COMP(data, form_x, form_y, lasso = lasso);

ate.logit.s   <- ATE(net_tfa, e401, cond.comp$my_z1x, cond.comp$my_z0x, cond.comp$mz_x.logit);
att.logit.s   <- ATT(net_tfa, e401, cond.comp$my_z0x, cond.comp$mz_x.logit, cond.comp$mz);

se.ate.logit.s   <- SE.ATE(net_tfa, e401, cond.comp$my_z1x, cond.comp$my_z0x, cond.comp$mz_x.logit);
se.att.logit.s   <- SE.ATT(net_tfa, e401, cond.comp$my_z0x, cond.comp$mz_x.logit, cond.comp$mz);



# Report results;

table <- matrix(0, ncol = 4, nrow = 6, dimnames = list(c('OLS', 'ATE',  'ATT-Logit', 'OLS-LASSO', 'ATE-LASSO', 'ATT-LASSO'), c('Est.', 'Std. Error', '95% LCI','95% UCI')));

table[1,1] <- ate.ols;
table[1,2] <- se.ate.ols;
table[1,3] <- ate.ols - qnorm(1-alpha/2)*se.ate.ols;
table[1,4] <- ate.ols + qnorm(1-alpha/2)*se.ate.ols;

table[2,1] <- ate.logit;
table[2,2] <- se.ate.logit;
table[2,3] <- ate.logit - qnorm(1-alpha/2)*se.ate.logit;
table[2,4] <- ate.logit + qnorm(1-alpha/2)*se.ate.logit;

table[3,1] <- att.logit;
table[3,2] <- se.att.logit;
table[3,3] <- att.logit - qnorm(1-alpha/2)*se.att.logit;
table[3,4] <- att.logit + qnorm(1-alpha/2)*se.att.logit;

table[4,1] <- ate.ols.s;
table[4,2] <- se.ate.ols.s;
table[4,3] <- ate.ols.s - qnorm(1-alpha/2)*se.ate.ols.s;
table[4,4] <- ate.ols.s + qnorm(1-alpha/2)*se.ate.ols.s;

table[5,1] <- ate.logit.s;
table[5,2] <- se.ate.logit.s;
table[5,3] <- ate.logit.s - qnorm(1-alpha/2)*se.ate.logit.s;
table[5,4] <- ate.logit.s + qnorm(1-alpha/2)*se.ate.logit.s;

table[6,1] <- att.logit.s;
table[6,2] <- se.att.logit.s;
table[6,3] <- att.logit.s - qnorm(1-alpha/2)*se.att.logit.s;
table[6,4] <- att.logit.s + qnorm(1-alpha/2)*se.att.logit.s;


print(table);

xtable(table, digits=0, caption = "Treatment Effects", label = "teffects");


#------------------------------------------------------------------------------------#;
# 2 - Specification with two-way interactions;
#------------------------------------------------------------------------------------#;


# Linear regression;

form_y    <- "net_tfa";
form_z    <- "e401";
form_x    <- "(poly(age, 6) + poly(inc, 8) + poly(educ, 4) + poly(fsize,2) + as.factor(marr) + as.factor(twoearn) + as.factor(db) + as.factor(pira) + as.factor(hown))^2";



lm.fit.y   <- lm(paste(form_y, "~", form_x));
ry         <- lm.fit.y$res;

lm.fit.z    <- lm(paste(form_z, "~", form_x));
rz          <- lm.fit.z$res;

lm.fit.ry    <- lm(ry ~ rz - 1);
ate.ols      <- lm.fit.ry$coef;
HCV.coefs    <- vcovHC(lm.fit.ry, type = 'HC');
se.ate.ols   <- sqrt(diag(HCV.coefs)); # White std errors

# Linear regression with lasso selection;

form_y    <- "net_tfa";
form_z    <- "e401";
form_x    <- "(poly(age, 6) + poly(inc, 8) + poly(educ, 4) + poly(fsize,2) + as.factor(marr) + as.factor(twoearn) + as.factor(db) + as.factor(pira) + as.factor(hown))^2";


lm.fit.y        <- lm(paste(form_y, "~", form_x), x = TRUE, y = TRUE);
cv.lasso        <- cv.glmnet(lm.fit.y$x[ ,-1], lm.fit.y$y, family = "gaussian", nfolds = 500);
lasso.fit.y     <- glmnet(lm.fit.y$x[ ,-1], lm.fit.y$y, family = "gaussian", lambda = cv.lasso$lambda.1se);
selected        <- c(TRUE, !as.vector(lasso.fit.y$beta == 0));
plasso.fit.y    <- lm(lm.fit.y$y ~ lm.fit.y$x[ , selected]);
ry              <- plasso.fit.y$res;

lm.fit.z        <- lm(paste(form_z, "~", form_x), x = TRUE, y = TRUE);
cv.lasso        <- cv.glmnet(lm.fit.z$x[ ,-1], lm.fit.z$y, family = "gaussian", nfolds = 500);
lasso.fit.z     <- glmnet(lm.fit.z$x[ ,-1], lm.fit.z$y, family = "gaussian", lambda = cv.lasso$lambda.1se);
selected        <- c(TRUE, !as.vector(lasso.fit.z$beta == 0));
plasso.fit.z    <- lm(lm.fit.z$y ~ lm.fit.z$x[ , selected]);
rz              <- plasso.fit.z$res;


lm.fit.ry      <- lm(ry ~ rz - 1);
ate.ols.s      <- lm.fit.ry$coef;
HCV.coefs      <- vcovHC(lm.fit.ry, type = 'HC');
se.ate.ols.s   <- sqrt(diag(HCV.coefs)); # White std errors



# Nonparametric estimation;

form_y    <- "net_tfa";
form_x    <- "(poly(age, 6) + poly(inc, 8) + poly(educ, 4) + poly(fsize,2) + as.factor(marr) + as.factor(twoearn) + as.factor(db) + as.factor(pira) + as.factor(hown))^2";

lasso     <- FALSE;
cond.comp <- COND.COMP(data, form_x, form_y, lasso = lasso);

ate.logit   <- ATE(net_tfa, e401, cond.comp$my_z1x, cond.comp$my_z0x, cond.comp$mz_x.logit);
att.logit   <- ATT(net_tfa, e401, cond.comp$my_z0x, cond.comp$mz_x.logit, cond.comp$mz);

se.ate.logit   <- SE.ATE(net_tfa, e401, cond.comp$my_z1x, cond.comp$my_z0x, cond.comp$mz_x.logit);
se.att.logit   <- SE.ATT(net_tfa, e401, cond.comp$my_z0x, cond.comp$mz_x.logit, cond.comp$mz);

# Nonparametric estimation with lasso selection;

form_y    <- "net_tfa";
form_x    <- "(poly(age, 6) + poly(inc, 8) + poly(educ, 4) + poly(fsize,2) + as.factor(marr) + as.factor(twoearn) + as.factor(db) + as.factor(pira) + as.factor(hown))^2";


lasso     <- TRUE;
cond.comp <- COND.COMP(data, form_x, form_y, lasso = lasso);

ate.logit.s   <- ATE(net_tfa, e401, cond.comp$my_z1x, cond.comp$my_z0x, cond.comp$mz_x.logit);
att.logit.s   <- ATT(net_tfa, e401, cond.comp$my_z0x, cond.comp$mz_x.logit, cond.comp$mz);

se.ate.logit.s   <- SE.ATE(net_tfa, e401, cond.comp$my_z1x, cond.comp$my_z0x, cond.comp$mz_x.logit);
se.att.logit.s   <- SE.ATT(net_tfa, e401, cond.comp$my_z0x, cond.comp$mz_x.logit, cond.comp$mz);



# Report results;

table <- matrix(0, ncol = 4, nrow = 6, dimnames = list(c('OLS', 'ATE',  'ATT-Logit', 'OLS-LASSO', 'ATE-LASSO', 'ATT-LASSO'), c('Est.', 'Std. Error', '95% LCI','95% UCI')));

table[1,1] <- ate.ols;
table[1,2] <- se.ate.ols;
table[1,3] <- ate.ols - qnorm(1-alpha/2)*se.ate.ols;
table[1,4] <- ate.ols + qnorm(1-alpha/2)*se.ate.ols;

table[2,1] <- ate.logit;
table[2,2] <- se.ate.logit;
table[2,3] <- ate.logit - qnorm(1-alpha/2)*se.ate.logit;
table[2,4] <- ate.logit + qnorm(1-alpha/2)*se.ate.logit;

table[3,1] <- att.logit;
table[3,2] <- se.att.logit;
table[3,3] <- att.logit - qnorm(1-alpha/2)*se.att.logit;
table[3,4] <- att.logit + qnorm(1-alpha/2)*se.att.logit;

table[4,1] <- ate.ols.s;
table[4,2] <- se.ate.ols.s;
table[4,3] <- ate.ols.s - qnorm(1-alpha/2)*se.ate.ols.s;
table[4,4] <- ate.ols.s + qnorm(1-alpha/2)*se.ate.ols.s;

table[5,1] <- ate.logit.s;
table[5,2] <- se.ate.logit.s;
table[5,3] <- ate.logit.s - qnorm(1-alpha/2)*se.ate.logit.s;
table[5,4] <- ate.logit.s + qnorm(1-alpha/2)*se.ate.logit.s;

table[6,1] <- att.logit.s;
table[6,2] <- se.att.logit.s;
table[6,3] <- att.logit.s - qnorm(1-alpha/2)*se.att.logit.s;
table[6,4] <- att.logit.s + qnorm(1-alpha/2)*se.att.logit.s;


print(table);

xtable(table, digits=0, caption = "Treatment Effects", label = "teffects");


