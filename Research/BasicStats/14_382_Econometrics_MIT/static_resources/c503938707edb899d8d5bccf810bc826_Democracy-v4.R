#################################################################################################
#  This is an empirical application of lineal panel data methods to the effect of democracy
#  on economic growth based on Acemoglu, Naidu, Restrepo and Robinson (2005), "Democracy Does
#  Cause Growth," forthcoming JPE
#
#  14.382 L8 MIT.  V. Chernozhukov and I. Fernandez-Val
#
# Data source: Daron Acemoglu (MIT), N = 147 countries, T = 23 years (1987-2009)
# include data for four lags of dependent variable, balanced panel
#
# Description of the data: the sample selection and variable contruction follow
#
# The variables in the data set include:
#
# country_name    = Country name
# wbcode          = World Bank country code
# year            = Year 
# id              = Generated numeric country code
# dem             = Democracy measure by ANRR
# lgdp            = log of GDP per capita in 2000 USD from World Bank

#################################################################################################


# Updated on 04/07/2016


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
library(readstata13);

data <- read.dta13("Data/Democracy/democracy-balanced-l4.dta")
data <- pdata.frame(data, index = c("id","year"));

attach(data);


########## Descriptive statistics

options(digits=2);
dstat <- cbind(sapply(data[, c(5:6)], mean), sapply(data[,c(5:6)], sd), apply(data[data$dem==1, c(5:6)],2, mean), apply(data[data$dem==0, c(5:6)], 2, mean));
dstat <- rbind(dstat, c(nrow(data), nrow(data), sum(data$dem==1), sum(data$dem==0)));
dimnames(dstat) <- list(c("Democracy", "Log(GDP)", "Number Obs."), c("Mean", "SD", "Dem = 1", "Dem = 0"));
xtable(dstat);

########## OLS estimation

form.ols <- lgdp ~ dem + lag(lgdp, 1:4) + factor(year) -1 ;

ols.fit   <- plm(form.ols, data, model = "pooling", index = c("id","year"));
coefs.ols <- coef(ols.fit); 
se.ols    <- summary(ols.fit)$coef[ ,2];
HCV.coefs <- vcovHC(ols.fit, method = 'arellano', type = 'HC0', cluster = 'group');
cse.ols   <- sqrt(diag(HCV.coefs)); # Clustered std errors


########## Fixed Effects estimation

form.fe <- lgdp ~ dem + lag(lgdp, 1:4) - 1;

fe.fit    <- plm(form.fe, data, model = "within", effect = "twoways", index = c("id","year"));
coefs.fe  <- coef(fe.fit); 
se.fe     <- summary(fe.fit)$coef[ ,2];
HCV.coefs <- vcovHC(fe.fit, cluster = 'group');
cse.fe    <- sqrt(diag(HCV.coefs)); # Clustered std errors

########## First Differences estimation

form.fd <- lgdp ~ dem  + lag(lgdp, 1:4) +   
        I(year==1993) + I(year==1994) + I(year==1995) + I(year==1996) + 
        I(year==1997) + I(year==1998) + I(year==1999) + I(year==2000) +  
        I(year==2001) + I(year==2002) + I(year==2003) + I(year==2004) +
        I(year==2005) + I(year==2006) + I(year==2007) + I(year==2008) + I(year==2009);


fd.fit    <- plm(form.fd, data, model = "fd", index = c("id","year"));
coefs.fd  <- coef(fd.fit); 
se.fd     <- summary(fd.fit)$coef[ ,2];
HCV.coefs <- vcovHC(fd.fit, method = 'arellano', type = 'HC0', cluster = 'group');
cse.fd    <- sqrt(diag(HCV.coefs)); # Clustered std errors

########## Arellano-Bond estimation 


form.ab <- lgdp ~ dem + lag(lgdp, 1:4) | lag(lgdp, 2:99) + lag(dem, 1:99);
ab.fit <- pgmm(form.ab, data, model = "twosteps", effect = "twoways", robust = TRUE );
coefs.ab  <- coef(ab.fit); 
# se.ab     <- summary(ab.fit)$coef[ ,2];
HCV.coefs <- vcovHC(ab.fit, cluster = 'group');
cse.ab    <- sqrt(diag(HCV.coefs)); # Clustered std errors
Jtest.ab  <- sargan(ab.fit)$statistic;
Jdof.ab  <- sargan(ab.fit)$parameter;

########## Anderson-Hsiao estimation 


form.ah <- lgdp ~ dem + lag(lgdp, 1:4) | lag(lgdp, 2) + lag(dem, 1);
ah.fit <- pgmm(form.ah, data, model = "twosteps", effect = "twoways", robust = TRUE);
coefs.ah  <- coef(ah.fit); 
# se.ah     <- summary(ah.fit)$coef[ ,2];
HCV.coefs <- vcovHC(ah.fit, cluster = 'group');
cse.ah    <- sqrt(diag(HCV.coefs)); # Clustered std errors
Jtest.ah  <- sargan(ah.fit)$statistic;
Jdof.ah  <- sargan(ah.fit)$parameter;





########## Panel bootstrap std errors

set.seed(8);
N <- length(unique(id));
T <- length(unique(year));
R <- 500;
coefs.ols.b <- matrix(0, ncol = length(coefs.ols), nrow = R);
coefs.fe.b  <- matrix(0, ncol = length(coefs.fe), nrow = R);
coefs.fd.b  <- matrix(0, ncol = length(coefs.fd), nrow = R);
coefs.ab.b  <- matrix(0, ncol = length(coefs.ab), nrow = R);
coefs.ah.b  <- matrix(0, ncol = length(coefs.ah), nrow = R);


for (r in 1:R) {;
                ids                <- kronecker(sample.int(N, N, replace = TRUE), rep(1,T));
                data.b             <- data[(ids-1)*T + rep(c(1:T),N), ];
                data.b$id          <- kronecker(c(1:N), rep(1,T));
                data.b$year        <- rep(c(1987:2009),N);
                data.b             <- data.frame(data.b);
                data.b             <- pdata.frame(data.b, index = c("id","year"));         # reset indexes of the panel                        
                coefs.ols.b[r, ]   <- coef(plm(form.ols, data.b, model = "pooling", index = c("id","year")));
                coefs.fe.b[r, ]    <- coef(plm(form.fe, data.b, model = "within", effect = "twoways", index = c("id","year")));
                coefs.fd.b[r, ]    <- coef(plm(form.fd, data.b, model = "fd", index = c("id","year")));  
                coefs.ab.b[r, ]    <- coef(pgmm(form.ab, data.b, model = "twosteps", effect = "twoways" ));                  
                coefs.ah.b[r, ]    <- coef(pgmm(form.ah, data.b, model = "twosteps", effect = "twoways" ));                                  
};

bse.ols <- apply(coefs.ols.b, 2, sd);
bse.fe  <- apply(coefs.fe.b, 2, sd);
bse.fd  <- apply(coefs.fd.b, 2, sd);
bse.ab  <- apply(coefs.ab.b, 2, sd);
bse.ah  <- apply(coefs.ah.b, 2, sd);

######## Table of results;

options(digits=3);
table.all <- matrix(NA, nrow = 18, ncol = 5, dimnames = list(c("Democracy", "CSE", "BSE", "L1.log(gdp)",  "CSE1", "BSE1", "L2.log(gdp)",  "CSE2", "BSE2","L3.log(gdp)",  "CSE3", "BSE3", "L4.log(gdp)",  "CSE4", "BSE4","J-test", "p-val","dof"), c("Pooled", "FD", "FE", "GMM-FD1", "GMM-FD2")));

table.all[c(1,4,7,10,13), 1] <- coefs.ols[1:5];
table.all[c(2,5,8,11,14), 1] <- cse.ols[1:5];
table.all[c(3,6,9,12,15), 1] <- bse.ols[1:5];

table.all[c(1,4,7,10,13), 2] <- coefs.fd[2:6];
table.all[c(2,5,8,11,14), 2] <- cse.fd[2:6];
table.all[c(3,6,9,12,15), 2] <- bse.fd[2:6];

table.all[c(1,4,7,10,13), 3] <- coefs.fe[1:5];
table.all[c(2,5,8,11,14), 3] <- cse.fe[1:5];
table.all[c(3,6,9,12,15), 3] <- bse.fe[1:5];

table.all[c(1,4,7,10,13), 4] <- coefs.ah[1:5];
table.all[c(2,5,8,11,14), 4] <- cse.ah[1:5];
table.all[c(3,6,9,12,15), 4] <- bse.ah[1:5];

table.all[c(1,4,7,10,13), 5] <- coefs.ab[1:5];
table.all[c(2,5,8,11,14), 5] <- cse.ab[1:5];
table.all[c(3,6,9,12,15), 5] <- bse.ab[1:5];


table.all[1, ] <- 100 * table.all[1, ];
table.all[2, ] <- 100 * table.all[2, ];
table.all[3, ] <- 100 * table.all[3, ];


table.all[16 ,4] <- Jtest.ah;
table.all[18 ,4] <- as.integer(Jdof.ah);
table.all[17 ,4] <- 1 - pchisq(Jtest.ab, Jdof.ah);

table.all[16 ,5] <- Jtest.ab;
table.all[18 ,5] <- as.integer(Jdof.ab);
table.all[17 ,5] <- 1 - pchisq(Jtest.ab, Jdof.ab);



xtable(table.all, digits=2);



# Long run effects;


lr.fe <- 100*coefs.fe[1]/(1 - sum(coefs.fe[2:5]));
lr.ab <- 100*coefs.ab[1]/(1 - sum(coefs.ab[2:5]));
lr.ah <- 100*coefs.ah[1]/(1 - sum(coefs.ah[2:5]));

print(c(lr.fe, lr.ab, lr.ah));