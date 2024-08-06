## ----include=FALSE------------------------------------------------------------
##### _Advanced Data Analysis from an Elementary Point of View_ #####
# R code for the chapter "Mixture Models"
# Please do not re-distribute or use without attribution
# http://www.stat.cmu.edu/~cshalizi/ADAfaEPoV/




## ----log-is-concave, echo=FALSE, out.width="0.5\\textwidth"-------------------
curve(log(x),from=0.4,to=2.1)
segments(0.5,log(0.5),2,log(2),lty=2)

## ----eval=FALSE---------------------------------------------------------------
## curve(log(x),from=0.4,to=2.1)
## segments(0.5,log(0.5),2,log(2),lty=2)


## -----------------------------------------------------------------------------
snoqualmie <- scan("http://www.stat.washington.edu/peter/book.data/set1",skip=1)
snoq <- snoqualmie[snoqualmie > 0]


## -----------------------------------------------------------------------------
summary(snoq)


## ----snoq-histogram, echo=FALSE, fig.keep="last"------------------------------
plot(hist(snoq,breaks=101),col="grey",border="grey",freq=FALSE,
     xlab="Precipitation (1/100 inch)",main="Precipitation in Snoqualmie Falls")
lines(density(snoq),lty="dashed")

## ----eval=FALSE---------------------------------------------------------------
## plot(hist(snoq,breaks=101),col="grey",border="grey",freq=FALSE,
##      xlab="Precipitation (1/100 inch)",main="Precipitation in Snoqualmie Falls")
## lines(density(snoq),lty="dashed")


## ----message=FALSE, results="hide"--------------------------------------------
library(mixtools)
snoq.k2 <- normalmixEM(snoq,k=2,maxit=100,epsilon=0.01)


## ----results="hide"-----------------------------------------------------------
snoq.k2 <- normalmixEM(snoq,k=2,maxit=100,epsilon=0.01)


## -----------------------------------------------------------------------------
summary(snoq.k2)


## -----------------------------------------------------------------------------
# Plot the (scaled) density associated with a Gaussian cluster
# Inputs: mixture object (mixture)
  # index number of the cluster (cluster.number)
  # optional additional arguments to curve (...)
# Outputs: None useful
# Side-effects: Plot is added to the current display
plot.gaussian.clusters <- function(mixture, cluster.number, ...) {
  curve(mixture$lambda[cluster.number] *
        dnorm(x,mean=mixture$mu[cluster.number],
        sd=mixture$sigma[cluster.number]), add=TRUE, ...)
}


## ----snoq-plus-two-modes,echo=FALSE, fig.keep="last"--------------------------
plot(hist(snoq,breaks=101),col="grey",border="grey",freq=FALSE,
     xlab="Precipitation (1/100 inch)",main="Precipitation in Snoqualmie Falls")
lines(density(snoq),lty=2)
invisible(sapply(1:2,plot.gaussian.clusters,mixture=snoq.k2))

## ----eval=FALSE---------------------------------------------------------------
## plot(hist(snoq,breaks=101),col="grey",border="grey",freq=FALSE,
##      xlab="Precipitation (1/100 inch)",main="Precipitation in Snoqualmie Falls")
## lines(density(snoq),lty=2)
## invisible(sapply(1:2,plot.gaussian.clusters,mixture=snoq.k2))


## -----------------------------------------------------------------------------
pnormmix <- function(x,mixture) {
  lambda <- mixture$lambda
  k <- length(lambda)
  pnorm.from.mix <- function(x,cluster) {
    lambda[cluster]*pnorm(x,mean=mixture$mu[cluster],
                            sd=mixture$sigma[cluster])
  }
  pnorms <- sapply(1:k,pnorm.from.mix,x=x)
  return(rowSums(pnorms))
}


## ----pp-plot, echo=FALSE------------------------------------------------------
distinct.snoq <- sort(unique(snoq))
tcdfs <- pnormmix(distinct.snoq,mixture=snoq.k2)
ecdfs <- ecdf(snoq)(distinct.snoq)
plot(tcdfs,ecdfs,xlab="Theoretical CDF",ylab="Empirical CDF",xlim=c(0,1),
     ylim=c(0,1))
abline(0,1)

## ----eval=FALSE---------------------------------------------------------------
## distinct.snoq <- sort(unique(snoq))
## tcdfs <- pnormmix(distinct.snoq,mixture=snoq.k2)
## ecdfs <- ecdf(snoq)(distinct.snoq)
## plot(tcdfs,ecdfs,xlab="Theoretical CDF",ylab="Empirical CDF",xlim=c(0,1),
##      ylim=c(0,1))
## abline(0,1)


## -----------------------------------------------------------------------------
# Probability density corresponding to a Gaussian mixture model
# Inputs: location for evaluating the pdf (x)
  # mixture-model object (mixture)
  # whether or not output should be logged (log)
# Output: the (possibly logged) PDF at the point(s) x
dnormalmix <- function(x,mixture,log=FALSE) {
  lambda <- mixture$lambda
  k <- length(lambda)
  # Calculate share of likelihood for all data for one cluster
  like.cluster <- function(x,cluster) {
    lambda[cluster]*dnorm(x,mean=mixture$mu[cluster],
                            sd=mixture$sigma[cluster])
  }
  # Create array with likelihood shares from all clusters over all data
  likes <- sapply(1:k,like.cluster,x=x)
  # Add up contributions from clusters
  d <- rowSums(likes)
  if (log) {
    d <- log(d)
  }
  return(d)
}

# Evaluate the loglikelihood of a mixture model at a vector of points
# Inputs: vector of data points (x)
  # mixture model object (mixture)
# Output: sum of log probability densities over the points in x
loglike.normalmix <- function(x,mixture) {
  loglike <- dnormalmix(x,mixture,log=TRUE)
  return(sum(loglike))
}


## -----------------------------------------------------------------------------
loglike.normalmix(snoq,mixture=snoq.k2)


## ----results="hide"-----------------------------------------------------------
n <- length(snoq)
data.points <- 1:n
data.points <- sample(data.points) # Permute randomly
train <- data.points[1:floor(n/2)] # First random half is training
test <- data.points[-(1:floor(n/2))] # 2nd random half is testing
candidate.cluster.numbers <- 2:10
loglikes <- vector(length=1+length(candidate.cluster.numbers))
# k=1 needs special handling
mu<-mean(snoq[train]) # MLE of mean
sigma <- sd(snoq[train])*sqrt((n-1)/n) # MLE of standard deviation
loglikes[1] <- sum(dnorm(snoq[test],mu,sigma,log=TRUE))
for (k in candidate.cluster.numbers) {
  mixture <- normalmixEM(snoq[train],k=k,maxit=400,epsilon=1e-2)
  loglikes[k] <- loglike.normalmix(snoq[test],mixture=mixture)
}


## -----------------------------------------------------------------------------
loglikes


## ----cv-loglikes, echo=FALSE, out.width="0.5\\textwidth"----------------------
plot(x=1:10, y=loglikes,xlab="Number of mixture clusters",
     ylab="Log-likelihood on testing data")

## ----eval=FALSE---------------------------------------------------------------
## plot(x=1:10, y=loglikes,xlab="Number of mixture clusters",
##      ylab="Log-likelihood on testing data")


## ----snoq-k9-1, include=FALSE-------------------------------------------------
snoq.k9 <- normalmixEM(snoq,k=9,maxit=400,epsilon=1e-2)

## ----snoq-k9-2, echo=FALSE, fig.keep="last"-----------------------------------
plot(hist(snoq,breaks=101),col="grey",border="grey",freq=FALSE,
     xlab="Precipitation (1/100 inch)",main="Precipitation in Snoqualmie Falls")
lines(density(snoq),lty=2)
invisible(sapply(1:9,plot.gaussian.clusters,mixture=snoq.k9))

## ----eval=FALSE---------------------------------------------------------------
## snoq.k9 <- normalmixEM(snoq,k=9,maxit=400,epsilon=1e-2)
## plot(hist(snoq,breaks=101),col="grey",border="grey",freq=FALSE,
##      xlab="Precipitation (1/100 inch)",main="Precipitation in Snoqualmie Falls")
## lines(density(snoq),lty=2)
## invisible(sapply(1:9,plot.gaussian.clusters,mixture=snoq.k9))


## ----pp-plot-k9, echo=FALSE---------------------------------------------------
distinct.snoq <- sort(unique(snoq))
tcdfs <- pnormmix(distinct.snoq,mixture=snoq.k9)
ecdfs <- ecdf(snoq)(distinct.snoq)
plot(tcdfs,ecdfs,xlab="Theoretical CDF",ylab="Empirical CDF",xlim=c(0,1),
     ylim=c(0,1))
abline(0,1)

## ----eval=FALSE---------------------------------------------------------------
## distinct.snoq <- sort(unique(snoq))
## tcdfs <- pnormmix(distinct.snoq,mixture=snoq.k9)
## ecdfs <- ecdf(snoq)(distinct.snoq)
## plot(tcdfs,ecdfs,xlab="Theoretical CDF",ylab="Empirical CDF",xlim=c(0,1),
##      ylim=c(0,1))
## abline(0,1)


## ----cluster-attributes-k9, echo=FALSE----------------------------------------
plot(0,xlim=range(snoq.k9$mu),ylim=range(snoq.k9$sigma),type="n",
     xlab="Cluster mean", ylab="Cluster standard deviation")
points(x=snoq.k9$mu,y=snoq.k9$sigma,pch=as.character(1:9),
       cex=sqrt(0.5+5*snoq.k9$lambda))

## ----eval=FALSE---------------------------------------------------------------
## plot(0,xlim=range(snoq.k9$mu),ylim=range(snoq.k9$sigma),type="n",
##      xlab="Cluster mean", ylab="Cluster standard deviation")
## points(x=snoq.k9$mu,y=snoq.k9$sigma,pch=as.character(1:9),
##        cex=sqrt(0.5+5*snoq.k9$lambda))


## ----kde-vs-k9, echo=FALSE----------------------------------------------------
plot(density(snoq),lty=2,ylim=c(0,0.04),
     main=paste("Comparison of density estimates\n",
                "Kernel vs. Gaussian mixture"),
     xlab="Precipitation (1/100 inch)")
curve(dnormalmix(x,snoq.k9),add=TRUE)

## ----eval=FALSE---------------------------------------------------------------
## plot(density(snoq),lty=2,ylim=c(0,0.04),
##      main=paste("Comparison of density estimates\n",
##                 "Kernel vs. Gaussian mixture"),
##      xlab="Precipitation (1/100 inch)")
## curve(dnormalmix(x,snoq.k9),add=TRUE)


## -----------------------------------------------------------------------------
day.classes <- apply(snoq.k9$posterior,1,which.max)


## -----------------------------------------------------------------------------
snoqualmie.classes <- data.frame(precip=snoqualmie, class=0)
years <- 1948:1983
snoqualmie.classes$day <- rep(c(1:366,1:365,1:365,1:365),times=length(years)/4)
wet.days <- (snoqualmie > 0)
snoqualmie.classes$class[wet.days] <- day.classes


## -----------------------------------------------------------------------------
snoqualmie.classes$class[wet.days] <- snoq.k9$mu[day.classes]


## ----classes-vs-day-of-year,echo=FALSE,dev="png"------------------------------
plot(x=snoqualmie.classes$day, y=snoqualmie.classes$class,
     xlim=c(1,366),ylim=range(snoq.k9$mu),xaxt="n",
     xlab="Day of year",ylab="Expected precipiation (1/100 inch)",
     pch=16,cex=0.2)
axis(1,at=1+(0:11)*30)

## ----eval=FALSE---------------------------------------------------------------
## plot(x=snoqualmie.classes$day, y=snoqualmie.classes$class,
##      xlim=c(1,366),ylim=range(snoq.k9$mu),xaxt="n",
##      xlab="Day of year",ylab="Expected precipiation (1/100 inch)",
##      pch=16,cex=0.2)
## axis(1,at=1+(0:11)*30)


## ----bootcomp-histogram, echo=FALSE,results="hide"----------------------------
snoq.boot <- boot.comp(snoq,max.comp=10,mix.type="normalmix",
                       maxit=400,epsilon=1e-2)

## ----eval=FALSE---------------------------------------------------------------
## snoq.boot <- boot.comp(snoq,max.comp=10,mix.type="normalmix",
##                        maxit=400,epsilon=1e-2)


## -----------------------------------------------------------------------------
str(snoq.boot)

