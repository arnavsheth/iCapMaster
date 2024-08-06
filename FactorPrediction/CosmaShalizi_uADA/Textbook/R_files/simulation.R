## ----include=FALSE------------------------------------------------------------
##### _Advanced Data Analysis from an Elementary Point of View_ #####
# R code for the chapter "Simulation"
# Please do not re-distribute or use without attribution
# http://www.stat.cmu.edu/~cshalizi/ADAfaEPoV/






















## -----------------------------------------------------------------------------
# Generate a sequence of IID random variables, each in 1:k
# Inputs: number of variables to produce (n)
  # vector of probabilities of length k (prob)
# Output: vector of random numbers
rmultinoulli <- function(n,prob) {
  k <- length(prob)
  return(sample(1:k,size=n,replace=TRUE,prob=prob))
}










## -----------------------------------------------------------------------------
library(MASS)
data(geyser)
fit.ols <- lm(waiting~duration,data=geyser)


## ----real-geyser-plus-line,echo=FALSE-----------------------------------------
plot(geyser$duration,geyser$waiting,xlab="duration",ylab="waiting")
abline(fit.ols)



## -----------------------------------------------------------------------------
rgeyser <- function() {
  n <- nrow(geyser)
  sigma <- summary(fit.ols)$sigma
  new.waiting <- rnorm(n,mean=fitted(fit.ols),sd=sigma)
  new.geyser <- data.frame(duration=geyser$duration,
    waiting=new.waiting)
  return(new.geyser)
}


## ----geyser-density,echo=FALSE------------------------------------------------
hist(geyser$waiting,freq=FALSE,xlab="waiting",main="",sub="",col="grey")
lines(hist(rgeyser()$waiting,plot=FALSE),freq=FALSE,lty="dashed")



## ----real-geyser-plus-sims,echo=FALSE-----------------------------------------
plot(geyser$duration,geyser$waiting,xlab="duration",ylab="waiting")
abline(fit.ols)
points(rgeyser(),pch=20,cex=0.5)

