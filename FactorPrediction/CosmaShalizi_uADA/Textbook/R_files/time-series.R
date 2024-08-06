## ----include=FALSE------------------------------------------------------------
##### _Advanced Data Analysis from an Elementary Point of View_ #####
# R code for the chapter "Time Series"
# Please do not re-distribute or use without attribution
# http://www.stat.cmu.edu/~cshalizi/ADAfaEPoV/




## ----include=FALSE------------------------------------------------------------
# Load the package for classic data sets
library(datasets)
data(lynx)


## ----logistic-map-defn--------------------------------------------------------
# Synthetic data set
  # Logistic map corrupted with observational noise
logistic.map <- function(x,r=4) { r*x*(1-x) }
logistic.iteration <- function(n, x.init, r=4){
  x <- vector(length=n)
  x[1] <- x.init
  for (i in 1:(n-1)) {
    x[i+1] <- logistic.map(x[i],r=r)
  }
  return(x)
}
x <- logistic.iteration(1000,x.init=runif(1))
y <- x+rnorm(1000,mean=0,sd=0.05)


## ----lynx-and-logistics, echo=FALSE-------------------------------------------
par(mfrow=c(1,2))
# plot all of the lynx series
plot(lynx)
# Plot the first part of the synthetic time series
plot(y[1:100],xlab="t",ylab=expression(y[t]),type="l")
par(mfrow=c(1,1))



## ----acfs, echo=FALSE---------------------------------------------------------
par(mfrow=c(1,2))
# autocorrelation functions
acf(lynx)
acf(y)
par(mfrow=c(1,1))



## ----design.matrix.from.ts, include=FALSE-------------------------------------
# Included here to create the next plot --- but will appear "publicly"
# below
design.matrix.from.ts <- function(ts,order,right.older=TRUE) {
  n <- length(ts)
  x <- ts[(order+1):n]
  for (lag in 1:order) {
    if (right.older) {
      x <- cbind(x,ts[(order+1-lag):(n-lag)])
    } else {
      x <- cbind(ts[(order+1-lag):(n-lag)],x)
    }
  }
  lag.names <- c("lag0",paste("lag",1:order,sep=""))
  if (right.older) {
    colnames(x) <- lag.names
  } else {
    colnames(x) <- rev(lag.names)
  }
  return(as.data.frame(x))
}


## ----iterates-plots,echo=FALSE------------------------------------------------
par(mfrow=c(1,2))
plot(lag0 ~ lag1,data=design.matrix.from.ts(lynx,1),xlab=expression(lynx[t]),
  ylab=expression(lynx[t+1]),pch=16)
plot(lag0 ~ lag1,data=design.matrix.from.ts(y,1),xlab=expression(y[t]),
  ylab=expression(y[t+1]),pch=16)
par(mfrow=c(1,1))



## -----------------------------------------------------------------------------
# Convert a time series into a data frame of lagged values
# Input: a time series, maximum lag to use, whether older values go on the right
  # or the left
# Output: a data frame with (order+1) columns, named lag0, lag1, ... , and
  # length(ts)-order rows
# Included here to create the next plot --- but will appear "publicly"
# below
design.matrix.from.ts <- function(ts,order,right.older=TRUE) {
  n <- length(ts)
  x <- ts[(order+1):n]
  for (lag in 1:order) {
    if (right.older) {
      x <- cbind(x,ts[(order+1-lag):(n-lag)])
    } else {
      x <- cbind(ts[(order+1-lag):(n-lag)],x)
    }
  }
  lag.names <- c("lag0",paste("lag",1:order,sep=""))
  if (right.older) {
    colnames(x) <- lag.names
  } else {
    colnames(x) <- rev(lag.names)
  }
  return(as.data.frame(x))
}


## ----additive-autorgressive-models--------------------------------------------
# Fit an additive autoregressive model
  # additive model fitting is outsourced to mgcv::gam, with splines
# Inputs: time series (x), order of autoregression (order)
# Output: fitted GAM object
aar <- function(ts,order) {
  stopifnot(require(mgcv))
  # Automatically generate a suitable data frame from the time series
  # and a formula to go along with it
  fit <- gam(as.formula(auto.formula(order)),
    data=design.matrix.from.ts(ts,order))
  return(fit)
}

# Generate formula for an autoregressive GAM of a specified order
# Input: order (integer)
# Output: a formula which looks like
  # "lag0 ~ s(lag1) + s(lag2) + ... + s(lagorder)"
auto.formula <- function(order) {
  inputs <- paste("s(lag",1:order,")",sep="",collapse="+")
    form <- paste("lag0 ~ ",inputs)
    return(form)
}


## ----include=FALSE------------------------------------------------------------
# Load the mgcv package, but suppress its extraneous start-up messages
library(mgcv)


## ----aar-for-logistic, echo=FALSE---------------------------------------------
# Plot successive values of y against each other
plot(lag0 ~ lag1,data=design.matrix.from.ts(y,1),xlab=expression(y[t]),
  ylab=expression(y[t+1]),pch=16)
# Add the linear regression (which would be the AR(1) model)
abline(lm(lag0~lag1,data=design.matrix.from.ts(y,1)),col="red")
# Fit a first-order nonparametric autoregression, add fitted values
yaar1 <- aar(y,order=1)
points(y[-length(y)],fitted(yaar1),col="blue")



## ----aar2-for-lynx------------------------------------------------------------
lynx.aar2 <- aar(lynx,2)


## ----lynx-aa2-partials,echo=FALSE---------------------------------------------
plot(lynx.aar2,pages=1)



## ----lynx-aa2-fitted, echo=FALSE----------------------------------------------
plot(lynx)
lines(1823:1934,fitted(lynx.aar2),lty="dashed")



## ----lynx-aar2-out-of-sample, echo=FALSE--------------------------------------
lynx.aar2b <- aar(lynx[1:80],2)
out.of.sample <- design.matrix.from.ts(lynx[-(1:78)],2)
lynx.preds <- predict(lynx.aar2b,newdata=out.of.sample)
plot(lynx)
lines(1823:1900,fitted(lynx.aar2b),lty="dashed")
lines(1901:1934,lynx.preds,col="grey")



## ----ar8-vs-aar1, echo=FALSE--------------------------------------------------
# Plot successive values of y against each other
plot(lag0 ~ lag1,data=design.matrix.from.ts(y,1),xlab=expression(y[t]),
  ylab=expression(y[t+1]),pch=16)
# Add the linear regression (which would be the AR(1) model)
abline(lm(lag0~lag1,data=design.matrix.from.ts(y,1)),col="red")
# Fit a first-order nonparametric autoregression, add fitted values
yaar1 <- aar(y,order=1)
points(y[-length(y)],fitted(yaar1),col="blue")
library(tseries)
yar8 <- arma(y,order=c(8,0))
points(y[-length(y)],fitted(yar8)[-1],col="red")



## ----cond-var-for-lynx--------------------------------------------------------
sq.res <- residuals(lynx.aar2)^2
lynx.condvar1 <- gam(sq.res ~ s(lynx[-(1:2)]))
lynx.condvar2 <- gam(sq.res ~ s(lag1)+s(lag2),
  data=design.matrix.from.ts(lynx,2))


## ----lynx-preds-with-error-bars,echo=FALSE------------------------------------
plot(lynx,ylim=c(-500,10000))
sd1 <- sqrt(fitted(lynx.condvar1))
lines(1823:1934,fitted(lynx.aar2)+2*sd1,col="grey")
lines(1823:1934,fitted(lynx.aar2)-2*sd1,col="grey")
lines(1823:1934,sd1,lty="dotted")



## ----rblockboot---------------------------------------------------------------
# Simple block bootstrap
# Inputs: time series (ts), block length, length of output
# Output: one resampled time series
# Presumes: ts is a univariate time series
rblockboot <- function(ts,block.length,len.out=length(ts)) {
  # chop up ts into blocks
  the.blocks <- as.matrix(design.matrix.from.ts(ts,block.length-1,
    right.older=FALSE))
    # look carefully at design.matrix.from.ts to see why we need the -1
  # How many blocks is that?
  blocks.in.ts <- nrow(the.blocks)
  # Sanity-check
  stopifnot(blocks.in.ts == length(ts) - block.length+1)
  # How many blocks will we need (round up)?
  blocks.needed <- ceiling(len.out/block.length)
  # Sample blocks with replacement
  picked.blocks <- sample(1:blocks.in.ts,size=blocks.needed,replace=TRUE)
  # put the blocks in the randomly-selected order
  x <- the.blocks[picked.blocks,]
  # convert from a matrix to a vector and return
    # need transpose because R goes from vectors to matrices and back column by
    # column, not row by row
  x.vec <- as.vector(t(x))
    # Discard uneeded extra observations at the end silently
  return(x.vec[1:len.out])
}


## ----lynx-vs-block-boot,echo=FALSE--------------------------------------------
plot(lynx)
lines(1821:1934, rblockboot(lynx,4),col="blue")



## ----gdp-per-capita-setup, echo=FALSE-----------------------------------------
library(pdfetch)
# Fetch real (inflation-adjusted) US per-capita gross domestic product
# ("chained 2012 dollars", i.e., no inflation adjustment for 2012)
gdppc.fred <- pdfetch_FRED("A939RX0Q048SBEA")
# This comes as a complicated data type, so break it down to a simple data
# frame, converting dates to a year with a decimal fraction
library(xts) # Functions needed for pdfetch's preferred format
library(lubridate) # Provides useful date-conversion functions
gdppc <- data.frame(year=decimal_date(index(gdppc.fred)),
                    y=as.numeric(gdppc.fred))

## ----gdp-per-capita, echo=FALSE-----------------------------------------------
plot(gdppc,log="y",type="l",ylab="GDP per capita (constant 2012 dollars)")



## ----gdp-with-exponential-trend, echo=FALSE-----------------------------------
plot(gdppc,log="y",type="l",ylab="GDP per capita (constant 2012 dollars)")
gdppc.exp <- lm(log(y) ~ year, data=gdppc)
beta0 <- exp(coefficients(gdppc.exp)[1])
beta <- coefficients(gdppc.exp)[2]
curve(beta0*exp(beta*x), lty="dashed", add=TRUE)



## ----gdp-residuals-from-exponential-trend, echo=FALSE-------------------------
plot(gdppc$year,residuals(gdppc.exp),xlab="year",
  ylab="logged fluctuation around trend",type="l",lty="dashed")



## ----gdp-with-spline-curve, echo=FALSE----------------------------------------
plot(gdppc,log="y",type="l",ylab="GDP per capita (constant 2012 dollars)")
gdppc.exp <- lm(log(y) ~ year, data=gdppc)
beta0 <- exp(coefficients(gdppc.exp)[1])
beta <- coefficients(gdppc.exp)[2]
curve(beta0*exp(beta*x), lty="dashed", add=TRUE)
gdp.spline <- fitted(gam(y~s(year), data=gdppc))
lines(gdppc$year,gdp.spline,lty="dotted")



## ----gdp-spline-curve-fluctuations, echo=FALSE--------------------------------
plot(gdppc$year,residuals(gdppc.exp),xlab="year",
  ylab="logged fluctuation around trend",type="l",lty="dashed")
lines(gdppc$year, log(gdppc$y/gdp.spline), xlab="year",
  ylab="logged fluctuations around trend", lty="dotted")



## ----first-difference-gdp-per-capita, echo=FALSE------------------------------
plot(gdppc$year[-1],diff(log(gdppc$y)),type="l",xlab="year",
  ylab="differenced log GDP per capita")



## ----employment-pop-ratio, echo=FALSE-----------------------------------------
epr.fred <- pdfetch_FRED("LNU02300000")
# Fetch the monthly US employment to population ratio of the US, in perecent,
  # without seasonal adjustment.
# Now convert it to a simple data frame
epr <- data.frame(year=decimal_date(index(epr.fred)),
                  epr=as.numeric(epr.fred))
# Truncate to 1990 forward, for illustrative purposes
epr <- epr[epr$year > 1989,]
plot(epr, ylab="Percent", ylim=c(50,70), main="Employment to population ratio",
     type="l")



## ----change-point-acf, echo=FALSE---------------------------------------------
par(mfrow=c(2,2))
before_crash <- (epr$year < 2009)
after_crash <- (epr$year >= 2009)
epr_before <- epr$epr[before_crash]
epr_after <- epr$epr[after_crash]
pre <- rnorm(sum(before_crash), mean(epr_before), sd(epr_before))
post <- rnorm(sum(after_crash), mean(epr_after), sd(epr_after))
change <- data.frame(year=epr$year,
                     epr=c(pre,post))
plot(change,ylab="", type="l")
acf(change$epr, lag.max=50, main="ACF of surrogate series")
acf(epr$epr, lag.max=50, main="ACF of actual data")
par(mfrow=c(1,1))



## ----gdppc.ma4, echo=FALSE----------------------------------------------------
gdppc.ma4 <- arma(x=residuals(gdppc.exp),order=c(0,4))
plot(gdppc$year,residuals(gdppc.exp),type="l",xlab="year",
  ylab="logged fluctuations in real US GDP per capita")
lines(gdppc$year,fitted(gdppc.ma4),col="grey",lwd=2)

