## ----include=FALSE------------------------------------------------------------
##### _Advanced Data Analysis from an Elementary Point of View_ #####
# R code for the chapter "Logistic Regression"
# Please do not re-distribute or use without attribution
# http://www.stat.cmu.edu/~cshalizi/ADAfaEPoV/




## ----include=FALSE------------------------------------------------------------
library(faraway) # for logit and ilogit functions


## ----code:sim.logistic--------------------------------------------------------
sim.logistic <- function(x, beta.0,beta,bind=FALSE) {
  require(faraway) # For accessible logit and inverse-logit functions
  linear.parts <- beta.0+(x%*%beta)
  y <- rbinom(nrow(x), size=1, prob=ilogit(linear.parts))
  if (bind) { return(cbind(x,y)) } else { return(y) }
}

plot.logistic.sim <- function(x, beta.0, beta, n.grid=50,
                              labcex=0.3, col="grey", ...) {
  grid.seq <- seq(from=-1,to=1,length.out=n.grid)
  plot.grid <- as.matrix(expand.grid(grid.seq,grid.seq))
  require(faraway)
  p <- matrix(ilogit(beta.0 + (plot.grid %*% beta )),nrow=n.grid)
  contour(x=grid.seq,y=grid.seq,z=p, xlab=expression(x[1]),
          ylab=expression(x[2]),main="",labcex=labcex,col=col)
  y <- sim.logistic(x,beta.0,beta,bind=FALSE)
  points(x[,1],x[,2],pch=ifelse(y==1,"+","-"),col=ifelse(y==1,"blue","red"))
  invisible(y)
}


## ----logistic-regression-sim-results, echo=FALSE, results="hide"--------------
x <- matrix(runif(n=50*2,min=-1,max=1),ncol=2)
par(mfrow=c(2,2))
plot.logistic.sim(x,beta.0=-0.1,beta=c(-0.2,0.2))
y.1 <- plot.logistic.sim(x,beta.0=-0.5,beta=c(-1,1))
plot.logistic.sim(x,beta.0=-2.5,beta=c(-5,5))
plot.logistic.sim(x,beta.0=-2.5e2,beta=c(-5e2,5e2))



## ----our-first-logistic-regression--------------------------------------------
df <- data.frame(y=y.1, x1=x[,1], x2=x[,2])
logr <- glm(y ~ x1 + x2, data=df, family="binomial")


## -----------------------------------------------------------------------------
summary(logr,digits=2,signif.stars=FALSE)


## -----------------------------------------------------------------------------
mean(ifelse(fitted(logr)<0.5,0,1) != df$y)


## ----our-first-gam------------------------------------------------------------
library(mgcv)
(gam.1 <- gam(y~s(x1)+s(x2), data=df, family="binomial"))




## ----gam-fit-to-logistic-simulation, out.width="0.49\\textwidth", echo=FALSE----
plot(gam.1,residuals=TRUE,pages=0)



## ----code:sim.fitted.logistic-------------------------------------------------
# Simulate a fitted logistic regression and return a new data frame
# Inputs: data frame (df), fitted model (mdl)
# Outputs: new data frame
# Presumes: df contains columns with names for the covariates of mdl
simulate.from.logr <- function(df, mdl) {
  probs <- predict(mdl, newdata=df, type="response")
  df$y <- rbinom(n=nrow(df), size=1, prob=probs)
  return(df)
}


## ----delta.deviance.sim-------------------------------------------------------
# Simulate from an estimated logistic model, and refit both the logistic
  # regression and a generalized additive model
# Hard-codes the formula; better code would be more flexible
# Inputs: data frame with covariates (df), fitted logistic model (mdl)
# Output: difference in deviances
# Presumes: df has columns names x.1 and x.2.
delta.deviance.sim <- function (df,mdl) {
  sim.df <- simulate.from.logr(df,mdl)
  GLM.dev <- glm(y~x1+x2,data=sim.df,family="binomial")$deviance
  GAM.dev <- gam(y~s(x1)+s(x2),data=sim.df,family="binomial")$deviance
  return(GLM.dev - GAM.dev)
}


## ----run.delta.deviance.sim---------------------------------------------------
(delta.dev.observed <- logr$deviance - gam.1$deviance)
delta.dev <- replicate(100,delta.deviance.sim(df,logr))
mean(delta.dev.observed <= delta.dev)


## ----diff-in-deviance-when-null-is-true, echo=FALSE---------------------------
hist(delta.dev, main="",
     xlab="Amount by which GAM fits better than logistic regression")
abline(v=delta.dev.observed,col="grey",lwd=4)



## ----snoqualmie-setup---------------------------------------------------------
# Read in the whole data set as one big vector, skipping the first line of
# the file (header information)
snoqualmie <- scan("http://www.stat.washington.edu/peter/book.data/set1",skip=1)
# Create a two-column data frame, today's precipitation vs. tomorrow's
snoq <- data.frame(tomorrow=c(tail(snoqualmie,-1),NA),
                   today=snoqualmie)
# Make the data more comprehensible by adding years and days within each year
# First, what years are we talking about?
years <- 1948:1983
# How many days to each year?
days.per.year <- rep(c(366,365,365,365),length.out=length(years))
# Add a "year" column
snoq$year <- rep(years, times=days.per.year)
# Add a day-within-the-year column
snoq$day <- rep(c(1:366,1:365,1:365,1:365),times=length(years)/4)
# Trim the last row to get rid of the NA
snoq <- snoq[-nrow(snoq),]


## ----fig:histogram, echo=FALSE------------------------------------------------
hist(snoqualmie,n=50,probability=TRUE,xlab="Precipitation (1/100 inch)")
rug(snoqualmie,col="grey")



## ----fig:scatterplot, echo=FALSE----------------------------------------------
plot(tomorrow~today,data=snoq,
     xlab="Precipitation today (1/100 inch)",
     ylab="Precipitation tomorrow (1/100 inch)",cex=0.1)
rug(snoq$today,side=1,col="grey")
rug(snoq$tomorrow,side=2,col="grey")



## ----snoq.logistic------------------------------------------------------------
snoq.logistic <- glm((tomorrow > 0) ~ today, data=snoq, family=binomial)




## ----data-vs-logistic-predictions, echo=FALSE---------------------------------
plot((tomorrow>0)~today,data=snoq,xlab="Precipitation today (1/100 inch)",
     ylab="Positive precipitation tomorrow?")
rug(snoq$today,side=1,col="grey")
data.plot <- data.frame(today=(0:500))
pred.bands <- function(mdl,data,col="black",mult=1.96) {
    preds <- predict(mdl,newdata=data,se.fit=TRUE)
    lines(data[,1],ilogit(preds$fit),col=col)
    lines(data[,1],ilogit(preds$fit+mult*preds$se.fit),col=col,lty="dashed")
    lines(data[,1],ilogit(preds$fit-mult*preds$se.fit),col=col,lty="dashed")
}
pred.bands(snoq.logistic,data.plot)



## ----data-vs-logistic-predictions-plus-spline, echo=FALSE---------------------
plot((tomorrow>0)~today,data=snoq,xlab="Precipitation today (1/100 inch)",
     ylab="Positive precipitation tomorrow?")
rug(snoq$today,side=1,col="grey")
data.plot <- data.frame(today=(0:500))
pred.bands(snoq.logistic,data.plot)
snoq.spline <- smooth.spline(x=snoq$today,y=(snoq$tomorrow>0))
lines(snoq.spline,col="red")



## ----data-vs-logistic-and-gam, echo=FALSE-------------------------------------
library(mgcv)
plot((tomorrow>0)~today,data=snoq,xlab="Precipitation today (1/100 inch)",
     ylab="Positive precipitation tomorrow?")
rug(snoq$today,side=1,col="grey")
pred.bands(snoq.logistic,data.plot)
lines(snoq.spline,col="red")
snoq.gam <- gam((tomorrow>0)~s(today),data=snoq,family=binomial)
pred.bands(snoq.gam,data.plot,"blue")



## ----snoq.sim-----------------------------------------------------------------
# Simulate from the fitted logistic regression model for Snoqualmie
# Presumes: fitted values of the model are probabilities.
snoq.sim <- function(model=snoq.logistic) {
  fitted.probs=fitted(model)
  return(rbinom(n=length(fitted.probs),size=1,prob=fitted.probs))
}




## -----------------------------------------------------------------------------
# Simulate from fitted logistic regression, re-fit logistic regression and
# GAM, calculate difference in deviances
diff.dev <- function(model=snoq.logistic,x=snoq[,"today"]) {
  y.new <- snoq.sim(model)
  GLM.dev <- glm(y.new ~ x,family=binomial)$deviance
  GAM.dev <- gam(y.new ~ s(x),family=binomial)$deviance
  return(GLM.dev-GAM.dev)
}




## -----------------------------------------------------------------------------
snoq2 <- data.frame(snoq,dry=ifelse(snoq$today==0,1,0))
snoq2.logistic <- glm((tomorrow > 0) ~ today + dry,data=snoq2,family=binomial)
snoq2.gam <- gam((tomorrow > 0) ~ s(today) + dry,data=snoq2,family=binomial)


## ----fig:snoq2, echo=FALSE----------------------------------------------------
plot((tomorrow>0)~today,data=snoq,xlab="Precipitation today (1/100 inch)",
  ylab="Positive precipitation tomorrow?")
rug(snoq$today,side=1,col="grey")
data.plot=data.frame(data.plot,dry=ifelse(data.plot$today==0,1,0))
lines(snoq.spline,col="red")
pred.bands(snoq2.logistic,data.plot)
pred.bands(snoq2.gam,data.plot,"blue")









## -----------------------------------------------------------------------------
frequency.vs.probability <- function(p.lower,p.upper=p.lower+0.01,
  model=snoq2.logistic,events=(snoq$tomorrow>0)) {
  fitted.probs <- fitted(model)
  indices <- (fitted.probs >= p.lower) & (fitted.probs < p.upper)
  ave.prob <- mean(fitted.probs[indices])
  frequency <- mean(events[indices])
  se <- sqrt(ave.prob*(1-ave.prob)/sum(indices))
  return(c(frequency=frequency,ave.prob=ave.prob,se=se))
}








## ----fig:calibration, echo=FALSE----------------------------------------------
plot(frequency~ave.prob,data=f.vs.p,xlim=c(0,1),ylim=c(0,1),
  xlab="Predicted probabilities",ylab="Observed frequencies")
rug(fitted(snoq2.logistic),col="grey")
abline(0,1,col="grey")
segments(x0=f.vs.p$ave.prob,y0=f.vs.p$ave.prob-1.96*f.vs.p$se,
  y1=f.vs.p$ave.prob+1.96*f.vs.p$se)

