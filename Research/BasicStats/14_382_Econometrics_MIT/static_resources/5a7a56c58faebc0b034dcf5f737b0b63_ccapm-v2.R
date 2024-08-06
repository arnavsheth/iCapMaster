# This empirical example estimates the CCAPM model of Hansen and Singleton (1982, ECMA) to illustrate the GMM estimation of 
# nonlinear models
# Authors: V. Chernozhukov and I. Fernandez-Val

# Data sources: Saint Louis Fed and Yahoo Finance
# URL: https://www.stlouisfed.org/, http://finance.yahoo.com/

#   Description of the data: monthly data from 1959:M01 to 2015:M01 (675 obs) of the following variables: 
# - PCEND: Personal Consumption Expenditures: Nondurable Goods (billions of dollars) from FED
# - PPCEND: Personal consumption expenditures: Nondurable goods (chain-type price index), DNDGRG3M086SBEA from FED
# - CNP16OV: Civilian Noninstitutional Population (thousands of persons) from FED
# - GS1: annualized 1-Year Treasury Constant Maturity Rate from FED
# - SP500: S&P 500 index at closing price of the first day of the month, ^GSPC from Yahoo Finance

# Updated on 03/06/2015

####################

# Part I: Processing Data

library(AER)
library(sandwich)
library(gmm)
library(quantmod)

# Reading the data
raw.data <- as.data.frame(read.csv("/Users/VC/Dropbox/Teaching/14.382/Data/CCAPM/ccapm-long.csv", header=T ))
attach(raw.data)

# Preparing data
rCpc        <- PCEND/(PPCEND * CNP16OV)
a.inflation <-  PPCEND/Lag(PPCEND, k=12) - 1

Rb          <- (1 + GS1/100 - a.inflation)^(1/12)   #total monthly return to bonds (deflated)
Rm          <- (SP500/Lag(SP500))*(Lag(PPCEND)/PPCEND)  #total monthly return to stocks (delated)
Rc          <- rCpc/Lag(rCpc)  #total monthly return to per-capita consumption (deflated)
Ret<-        na.omit(cbind(Rc, Rm, Rb))
colnames(Ret) <- c("Rc", "Rm", "Rb")

ts.plot(y, main="consumption, stock, and bond total returns", col=c(1:3))
detach("raw.data")



# Write out and Read-in Processed data

write.csv(Ret,file="/Users/VC/Dropbox/Teaching/14.382/Data/CCAPM/ccapm-ready-to-use.csv",row.names=FALSE)
Ret<- read.csv(file="/Users/VC/Dropbox/Teaching/14.382/Data/CCAPM/ccapm-ready-to-use.csv", header=T)
attach(Ret)




############  Score Functions #####################

# the score function here returns the n by m  matrix to be used in gmm package; 
# i.e. this is the m- vector of scores g(X_i, \thata) evaluated at each data point i=1,...,N) 

# the score function for estimationg CRRA preferences with 2 financials assets

g2.x.theta <- function(theta, x)
{
  y    <- x[ ,1:3]
  z    <- x[ ,-c(1:3)]
  return (cbind( (theta[1] * y[ ,2] * y[ ,1]^(-theta[2]) - 1)*z, 
                  (theta[1] * y[ ,3] * y[ ,1]^(-theta[2]) - 1)*z))
}



##################

# Instrument= 1 lag of consumption and stock returns
y     <- na.omit(cbind(Rc, Rb, Rm))
nlags <- 1
z    <- cbind(Lag(y[ ,1], c(1:nlags)), Lag(y[ ,2], c(1:nlags)), Lag(y[ ,3], c(1:nlags)))
x   <- na.omit(cbind(y,z))

# GMM fit
gmm2.fit  <- gmm(g2.x.theta, x, t0 = c(.99,3), type="iterative", vcov="iid")
print(summary(gmm2.fit))

# CUE fit
gmm2.cue.fit  <- gmm(g2.x.theta, x, t0 = c(1,0), type="cue", vcov="iid")
print(summary(gmm2.cue.fit))

# INSTRUMENT = 1 LAG + SQUARES AND INTERACTIONS 

y     <- na.omit(cbind(Rc, Rb, Rm))
nlags <- 1
z    <- cbind(Lag(y[ ,1], c(1:nlags)), Lag(y[ ,2], c(1:nlags)), Lag(y[ ,3], c(1:nlags)))
z<-  cbind( z[,1],z[,2],z[,3], z[,1]^2, z[,2]^2, z[,3]^2, z[,1]*z[,2], z[,1]*z[,3], z[,2]*z[,3])
x   <- na.omit(cbind(y,z))


# GMM fit
gmm2.zsq.fit  <- gmm(g2.x.theta, x, t0 = c(.99,1),  vcov="iid", type="iterative")
print(summary(gmm2.zsq.fit))

# CUE fit
gmm2.zsq.cue.fit  <- gmm(g2.x.theta, x, t0 = c(.99,1), type="cue", vcov="iid")
print(summary(gmm2.zsq.cue.fit))


######### Printing the results####################

library(xtable)  #package to generate latex tables


tableJ<- matrix(0, ncol=4, nrow=2)
tableJ[,1]<- summary(gmm2.fit)$stest[[2]]
tableJ[,2]<- summary(gmm2.cue.fit)$stest[[2]]
tableJ[,3]<- summary(gmm2.zsq.fit)$stest[[2]]
tableJ[,4]<- summary(gmm2.zsq.cue.fit)$stest[[2]]
colnames(tableJ)<- c("GMM-1", "CUE-1", "GMM-2", "CUE-2")
rownames(tableJ)<- c("J-statistic", "p-value")

tableE<- matrix(0, ncol=4, nrow=4);
tableE[c(1:2),1]<- summary(gmm2.fit)$coef[1,c(1:2)]
tableE[c(3:4),1]<- summary(gmm2.fit)$coef[2,c(1:2)]
tableE[c(1:2),2]<- summary(gmm2.cue.fit)$coef[1,c(1:2)]
tableE[c(3:4),2]<- summary(gmm2.cue.fit)$coef[2,c(1:2)]
tableE[c(1:2),3]<- summary(gmm2.zsq.fit)$coef[1,c(1:2)]
tableE[c(3:4),3]<- summary(gmm2.zsq.fit)$coef[2,c(1:2)]
tableE[c(1:2),4]<- summary(gmm2.zsq.cue.fit)$coef[1,c(1:2)]
tableE[c(3:4),4]<- summary(gmm2.zsq.cue.fit)$coef[2,c(1:2)]

colnames(tableE)<- c("GMM-1", "CUE-1", "GMM-2", "CUE-2")
rownames(tableE)<- c("estimated beta", "std error beta",  "estimated alpha", "std error for alpha")

#output tables in latex format

xtable(tableJ, digits=3, align=c(rep("c", 5)))
xtable(tableE, ditits=4, align=c(rep("c", 5)))


