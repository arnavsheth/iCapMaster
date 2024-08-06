rm(list=ls())
### Filepath
filepathIvan<-"/Users/ivan/Dropbox/Shared/14.382"
filepathVictor<-"/Users/VC/Dropbox/TEACHING/14.382"
filepathVira<-"/Users/virasemenova/Dropbox/14.382"
filepath<-filepathVira

#install.packages('boot')
library(boot)

### Used R functions (uncomment and run if unfamiliar):
# help(apply)
# help(get)
# help(boot)

# Set random number generator
set.seed(1)

############ GENERATE TRUE DATA
# Number of replications
R<-1000
# True data set size of each replication
N.true<-100
# Generate true data
data.true<-matrix(rexp(N.true*R),N.true,R)
# Statistic - function, defined by funname, to be applied to data
statistic<-function (data,funname) {apply(as.data.frame(data),2,get(funname))}

############ GENERATE FAKE DATA
# Fake data set size
N.fake<-100
data.fake<-rexp(N.fake)

############ COMPARE the DISTRIBUTION OF STATISTIC(TRUE DATA) and BOOTSTRAP STATISTIC
# Example 1: EBS working
funname ='mean';
# STATISTIC(TRUE DATA)
mean.draws<-statistic(data.true,'mean')
# BOOTSTRAP STATISTIC
boot.statistic<- function (data,indices) {statistic(data[indices],funname)}
# Number of bootstrap replications
R.boot<-1000
result.mean<-boot(data=data.fake,statistic=boot.statistic,R=R.boot)

# Example 2: EBS failing
funname ='min';
# STATISTIC(TRUE DATA)
min.draws<-statistic(data.true,'min')
# BOOTSTRAP STATISTIC
boot.statistic<- function (data,indices) {statistic(data[indices],funname)}

# Number of bootstrap replications
R.boot<-1000
result.min<-boot(data=data.fake,statistic=boot.statistic,R=R.boot)

## PLOTS
## Example 1: Mean

# Plot: true data
filename<-paste(filepath,"/Results/Sim_mean.eps",sep="") 
postscript(filename,horizontal=F, pointsize=15,width=8.0,height=8.0)
par(mfrow=c(2,1))
plot(hist(mean.draws,nclass=30,plot=FALSE), col="light blue", main="histogram for true draws", xlab="values of statistic")
se<- round(sqrt(var(mean.draws)), digits=4)
legend(1.2, 60, paste("se=", se))
# Plot: fake data
plot(hist(result.mean$t,nclass=30, plot=FALSE), col="blue", main="histogram for BS draws", xlab="values of statistic")
se<- round(sqrt(var(result.mean$t)), digits=4)
legend(1.1, 60, paste("se=", se))
dev.off()


# Example 2: Min
# Plot: true data
filename<-paste(filepath,"/Results/Sim_min1.eps",sep="") 
postscript(filename,horizontal=F, pointsize=15,width=8.0,height=8.0)
par(mfrow=c(2,1))
plot(hist(min.draws,nclass=30,plot=FALSE), col="light blue", main="histogram for true draws", xlab="values of statistic")
se<- round(sqrt(var(min.draws)), digits=4)
legend(.06, 100,  paste("se=", se))
# Plot: fake data
plot(hist(result.min$t,nclass=30, plot=FALSE), col="blue", main="histogram for BS draws", xlab="values of statistic")
se<- round(sqrt(var(result.min$t)), digits=4)
legend(.03, 300,  paste("se=", se))
dev.off()


### EBS in Hansen and Singleton (block BS is used)
install.packages("quantmod")
library(quantmod)

Ret<- read.csv(file=paste(filepath,"/Data/CCAPM/ccapm-ready-to-use.csv",sep=""), header=T)
attach(Ret)
y  <- na.omit(cbind(Rc, Rb, Rm))

# bootstrap GMM for technical instruments consisting of one lag

nlags<-1
z    <- cbind(Lag(y[ ,1], c(1:nlags)), Lag(y[ ,2], c(1:nlags)), Lag(y[ ,3], c(1:nlags)))
x   <- na.omit(cbind(y,z))

g2b.x.theta <- function(theta, x)
{
  y    <- x[ ,1:3]
  z    <- x[ ,-c(1:3)]
  return (cbind( (theta[1] * y[ ,2] * y[ ,1]^(-theta[2]) - 1)*z, 
                 (theta[1] * y[ ,3] * y[ ,1]^(-theta[2]) - 1)*z))
}

gmm.bs.statistic<- function(data,indices){ 
  x<- data[indices,]
  
  # GMM fit
  gmm2.coef <- gmm(g2b.x.theta, x, t0 = c(.99,.22), type="iterative", vcov="iid")$coef
  return(gmm2.coef)
}


gmm.ts.bs.statistic<- function(data){ 
  x<- data
  g2b.x.theta <- function(theta, x)
  {
    y    <- x[ ,1:3]
    z    <- x[ ,-c(1:3)]
    return (cbind( (theta[1] * y[ ,2] * y[ ,1]^(-theta[2]) - 1)*z, 
                   (theta[1] * y[ ,3] * y[ ,1]^(-theta[2]) - 1)*z))
  }
  # GMM fit
  gmm2.coef <- gmm(g2b.x.theta, x, t0 = c(.99,.22), type="iterative", vcov="iid")$coef
  return(gmm2.coef)
}


# bootstrap histograms and ses

library(sandwich)
library(gmm)
set.seed(1)
result.gmm.bs<- boot(data=x, statistic=gmm.bs.statistic,R=400)
options(digits=5)
ses<- round(sqrt(apply(result.gmm.bs$t, 2, var)),digits=4)


postscript(paste(filepath,"/Results/BS_GMM_indep.eps",sep=""),horizontal=F, pointsize=15,width=8.0,height=8.0)
par(mfrow=c(2,1))
plot(hist(result.gmm.bs$t[,1],nclass=23, plot=FALSE), col="light blue", main="histogram for BS draws", xlab="values of statistic")
legend(.999,20, c("se=", paste(ses[1])))
plot(hist(result.gmm.bs$t[,2],nclass=23, plot=FALSE), col="light blue", main="histogram for BS draws", xlab="values of statistic")
legend(.5,20, c("se=", paste(ses[2])))
dev.off()


confint(gmm(g2.x.theta, x, t0 = c(.99,3), type="iterative", vcov="iid"), level=.90)
quantile(result.gmm.bs$t[,1], c(0.05,.95))
quantile(result.gmm.bs$t[,2], c(.05,.95))


set.seed(1)
result.gmm.bs.ts<- tsboot(tseries=x, statistic=gmm.ts.bs.statistic, l=26, sim="fixed", R=200)
options(digits=2)
ses.ts<- round(sqrt(apply(result.gmm.bs.ts$t, 2, var)),digits=4)

postscript(paste(filepath,"/Results/BS_GMM_indep.eps",sep=""),horizontal=F, pointsize=15,width=8.0,height=8.0)
par(mfrow=c(2,1))
plot(hist(result.gmm.bs.ts$t[,1],nclass=23, plot=FALSE), col="light blue", main="histogram for BS draws", xlab="values of statistic")
legend(.999,15, c("se=", paste(ses.ts[1])))
plot(hist(result.gmm.bs.ts$t[,2],nclass=23, plot=FALSE), col="light blue", main="histogram for BS draws", xlab="values of statistic")
legend(.5,20, c("se=", paste(ses.ts[2])))
dev.off()

