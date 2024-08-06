## ----include=FALSE------------------------------------------------------------
##### _Advanced Data Analysis from an Elementary Point of View_ #####
# R code for the chapter "Bootstrap"
# Please do not re-distribute or use without attribution
# http://www.stat.cmu.edu/~cshalizi/ADAfaEPoV/




## ----rboot-bootstrap-bootstrap.se---------------------------------------------
# Generate random values of a statistic by repeatedly running a simulator
# Inputs: function to calculate the statistic (statistic)
  # function to run the simulation (simulator)
  # number of replicates (B)
# Output: array of bootstrapped values of the statistic, with B columns
  # To work more nicely with other functions, a vector is converted to an
  # array of dimensions 1*B
rboot <- function(statistic, simulator, B) {
  tboots <- replicate(B, statistic(simulator()))
  if(is.null(dim(tboots))) {
      tboots <- array(tboots, dim=c(1, B))
  }
  return(tboots)
}

# Summarize the sampling distribution of a statistic, obtained by repeatedly
  # running a simultor
# Inputs: array of bootstrapped statistics values (tboots)
  # function that summarizes the distribution (summarizer)
  # optional additional arguments to summarizer (...)
# Output: vector giving a summary of the statistic
# Presumes: tboots is an array with one column per simulation
  # each row of tboots is a separate component of the statistic
  # applying the summarizer to each row separately makes sense
bootstrap <- function(tboots, summarizer, ...) {
  summaries <- apply(tboots, 1, summarizer, ...)
  # using apply() like this has interchanged rows and columns
    # because each chunk processed by apply() results in a new column, but
    # here those chunks are the rows of tboots
  # therefore use transpose to restore original orientation
  return(t(summaries))
}

# Calculate a bootstrap standard error from scratch
# Inputs: function to calculate the statistic (statistic)
  # function to run the simulation (simulator)
  # number of replicates (B)
# Output: standard error for each
bootstrap.se <- function(statistic, simulator, B) {
    bootstrap(rboot(statistic, simulator, B), summarizer=sd)
}


## ----bootstrap.bias-----------------------------------------------------------
# Calculate bootstrap biases
# Inputs: function to run the simulation (simulator)
  # function to calculate the statistic (statistic)
  # number of replicates (B)
  # observed value of the statistic (t.hat)
# Outputs: difference between mean of replicates and observed value
bootstrap.bias <- function(simulator, statistic, B, t.hat) {
  # What's the expected value of the statistic, according to the bootstrap?
  expect <- bootstrap(rboot(statistic, simulator, B), summarizer=mean)
  # Bias is expected value minus truth
  return(expect-t.hat)
}


## ----bootstrap.ci-------------------------------------------------------------
# Find equal-tail interval with specified probability
# Inputs: vector of values to sort (x)
  # total tail probability (alpha)
# Output: length-two vector, giving interval of probability 1-alpha, with
  # probability alpha/2 in each tail
equitails <- function(x, alpha) {
  lower <- quantile(x, alpha/2)
  upper <- quantile(x, 1-alpha/2)
  return(c(lower, upper))
}

# Calculate (basic or pivotal) bootstrap confidence interval
# Inputs: function to calculate the statistic (statistic)
  # function to run the simulation (simulator)
  # optional array of bootstrapped values (tboots)
    # if this is not NULL, over-rides the statistic & simulator arguments
  # number of replicates (B)
  # observed value of statistic (t.hat)
  # confidence level (level)
# Outputs: two-column array with lower and upper confidence limits
bootstrap.ci <- function(statistic=NULL, simulator=NULL, tboots=NULL,
                         B=if(!is.null(tboots)) { ncol(tboots) },
                         t.hat, level) {
  # draw the bootstrap values, if not already provided
  if (is.null(tboots)) {
    # panic if we're not given an array of simulated values _and_ also lack
    # the means to calculate it for ourselves
    stopifnot(!is.null(statistic))
    stopifnot(!is.null(simulator))
    stopifnot(!is.null(B))
    tboots <- rboot(statistic, simulator, B)
  }
  # easier to work with error probability than confidence level
  alpha <- 1-level
  # Calculate probability intervals for each coordinate
  intervals <- bootstrap(tboots, summarizer=equitails, alpha=alpha)
  # Re-center the intervals around the observed values
  upper <- t.hat + (t.hat - intervals[,1])
  lower <- t.hat + (t.hat - intervals[,2])
  # calculate CIs, centered on observed value plus bootstrap fluctuations
    # around it
  CIs <- cbind(lower=lower, upper=upper)
  return(CIs)
}


## ----bootstrap-p-value--------------------------------------------------------
# Calculate a bootstrap p-value
# Inputs: function to calculate a test statistic (test)
  # function to run the simulation (simulator)
  # number of replicates (B)
  # observed value of the test statistic (testhat)
# Outputs: p-value for the hypothesis test
# Presumes: larger values of the test statistic are stronger evidence against
  # the null hypothesis
boot.pvalue <- function(test,simulator,B,testhat) {
  # bootstrap B values of the test statistic
  testboot <- rboot(B=B, statistic=test, simulator=simulator)
  # What proportion of simulated test statistics are at least as extreme as
    # the observed?
    # The +1 in numerator and denominator avoids the embarrassment of claiming
    # a p-value is exactly 0 on the basis of finite simulations
  p <- (sum(testboot >= testhat)+1)/(B+1)
  return(p)
}


## ----double-bootstrap---------------------------------------------------------
# Calculate a p-value by two levels of bootstrapping
  # Useful when an estimated parameter affects the distribution of the test
# Inputs: function to calculate a test statistic (test)
  # function to run the simulation (simulator)
  # number of replicates for top-level bootstrap (B1)
  # number of replicates per top-level replicate (B2)
  # function to estimate parameters (estimator)
  # estimate of parameter on actual data (thetahat)
  # observed value of the test statistic (testhat)
  # optional additional arguments to simulator (...)
# Outputs: p-value for the hypothesis test
# Presumes: larger values of the test statistic are stronger evidence against
  # the null hypothesis
  # simulator() can take thetahat as an argument
  # estimator() returns a value which simulator() can take as an argument
doubleboot.pvalue <- function(test, simulator, B1, B2, estimator, thetahat,
                              testhat, ...) {
  # For each top-level or outer replicate
  for (i in 1:B1) {
    # Run the simulator at the estimated parameter value
    xboot <- simulator(theta=thetahat, ...)
    # Re-estimate and re-calculate the test statistic
    thetaboot <- estimator(xboot)
    testboot[i] <- test(xboot)
    # Calculate a bootstrapped p-value for _that_ test
    pboot[i] <- boot.pvalue(test, simulator, B2, testhat=testboot[i],
                            theta=thetaboot)
  }
  # EXERCISE for the reader: replace that for() loop with something vectorized
  # Get an unadjusted p-value for our observed test statistic
  p <- (sum(testboot >= testhat)+1)/(B1+1)
  # How extreme is our un-adjusted p-value?
  p.adj <- (sum(pboot <= p)+1)/(B1+1)
  return(p.adj)
}


## ----pareto-example-setup-----------------------------------------------------
# Load the file with commands for the Pareto distribution
source("http://www.stat.cmu.edu/~cshalizi/ADAfaEPoV/code/pareto.R")
# Load the data on the wealth of the 400 richest Americans
wealth <- scan("http://www.stat.cmu.edu/~cshalizi/ADAfaEPoV/data/wealth.dat")
# Pareto distribution takes over after $9*10^8
x0 <- 9e8
# How many data points are there in that region?
n.tail <- sum(wealth >= x0)
# Fit the Pareto distribution by maximum likelihood
wealth.pareto <- pareto.fit(wealth, threshold=x0)


## ----wealth, echo=FALSE, results="hide"---------------------------------------
plot.survival.loglog(wealth, xlab="Net worth (dollars)",
  ylab="Fraction of top 400 above that worth")
rug(wealth, side=1, col="grey")
curve((n.tail/400)*ppareto(x, threshold=x0, exponent=wealth.pareto$exponent,
                           lower.tail=FALSE),
      add=TRUE, lty="dashed", from=x0, to=2*max(wealth))



## ----model-based-pareto-bootstrap---------------------------------------------
# Simulate the estimated Pareto distribution of wealth
# Inputs: none
# Output: vector of the same length as the original tail of the data
sim.wealth <- function() {
    rpareto(n=n.tail,
            threshold=wealth.pareto$xmin,
            exponent=wealth.pareto$exponent)
}

# Estimate the Pareto distribution with a fixed threshold and return the
  # estimate of the exponent
# Inputs: data vector (data)
# Output: estimated exponent
est.pareto <- function(data) { pareto.fit(data, threshold=x0)$exponent }


## ----pareto-uncertainties-----------------------------------------------------
pareto.se <- bootstrap.se(statistic=est.pareto, simulator=sim.wealth, B=1e4)
pareto.bias <- bootstrap.bias(statistic=est.pareto,
                              simulator=sim.wealth,
                              t.hat=wealth.pareto$exponent,
                              B=1e4)
pareto.ci <- bootstrap.ci(statistic=est.pareto, simulator=sim.wealth, B=1e4,
                          t.hat=wealth.pareto$exponent, level=0.95)


## ----pareto-ks----------------------------------------------------------------
# Calculate the KS goodness-of-fit statistic for a Pareto distribution,
  # using only obsevations above the Pareto's lower threshold
# Inputs: data vector (x)
  # theoretical exponent (exponent)
  # theoretical lower threshold (x0)
# Output: maximum distance between empirical CDF and theoretical CDF
ks.stat.pareto <- function(x, exponent, x0) {
  x <- x[x>=x0]
  ks <- ks.test(x, ppareto, exponent=exponent, threshold=x0)
  return(ks$statistic)
}

# Bootstrapped  KS goodness-of-fit p-value for a Pareto distribution,
  # using only obsevations above the Pareto's lower threshold
# Inputs: number of bootstrap replicate (B)
  # data vector (x)
  # theoretical exponent (exponent)
  # theoretical lower threshold (x0)
# Output: p-value for testing hypothesis that true CDF is the theoretical CDF
ks.pvalue.pareto <- function(B, x, exponent, x0) {
  testhat <- ks.stat.pareto(x, exponent, x0)
  testboot <- vector(length=B)
  # EXERCISE for the reader: replace the for() loop with replicate()
  for (i in 1:B) {
    xboot <- rpareto(length(x),exponent=exponent, threshold=x0)
    exp.boot <- pareto.fit(xboot,threshold=x0)$exponent
    testboot[i] <- ks.stat.pareto(xboot,exp.boot,x0)
  }
  p <- (sum(testboot >= testhat)+1)/(B+1)
  return(p)
}




## ----resample-----------------------------------------------------------------
# Resample a vector
  # That is, treat a sample as though it were a whole population, and draw
  # from it by sampling-with-replacement until we have a simulated data set
  # as big as the original
  # Equivalently, do IID draws from the empirical distribution
# Inputs: vector to resample (x)
# Outputs: vector of resampled values
resample <- function(x) { sample(x,size=length(x),replace=TRUE) }

# Resample whole rows from a data frame
  # That is, treat the rows as a population, and sample them with replacement
  # until we have a new data frame the same size as the original
  # Equivalently, draw IIDly from the joint empirical distribution over all
  # variables/columns
# Inputs: data frame to resample (data)
# Outputs: new data frame
resample.data.frame <- function(data) {
  # Resample the row indices
  sample.rows <- resample(1:nrow(data))
  # Return a new data frame with those rows in that order
  return(data[sample.rows,])
}


## ----resampling-pareto-CI-----------------------------------------------------
wealth.resample <- function() { resample(wealth[wealth >= x0]) }
pareto.CI.resamp <- bootstrap.ci(statistic=est.pareto,
                                 simulator=wealth.resample,
                                 t.hat=wealth.pareto$exponent,
                                 level=0.95, B=1e4)


## ----geyser-setup-------------------------------------------------------------
library(MASS)
data(geyser)
geyser.lm <- lm(waiting~duration,data=geyser)




## ----geyser.resample----------------------------------------------------------
# Deliberately uncommented
resample.geyser <- function() { resample.data.frame(geyser) }


## -----------------------------------------------------------------------------
# Estimate the geyser linear model on a data frame
# Inputs: data frame (data)
# Output: coefficient vector
# Presumes: data contains columns named "waiting" and "duration"
est.geyser.lm <- function(data) {
  fit <- lm(waiting ~ duration, data=data)
  return(coefficients(fit))
}


## ----geyser-confidence-intervals-by-resampling, tidy=FALSE--------------------
geyser.lm.ci <- bootstrap.ci(statistic=est.geyser.lm,
                             simulator=resample.geyser,
                             level=0.95,
                             t.hat=coefficients(geyser.lm),
                             B=1e4)




## ----geyser-kernel-regression-curves, results="hide"--------------------------
# Define a fixed grid of duration values for use in plotting model predctions
evaluation.points <- data.frame(duration=seq(from=0.8,
                                             to=5.5,
                                             length.out=200))

library(np)

# Fit a kernel regression of the sort appropriate to the geyser
# data, and return a vector of predicted values along a fixed grid
# Inputs: data frame (data)
  # optimization tolerance for bandwidth selection (tol, ftol)
  # data frame of values to predict at (plot.df)
# Output: vector of predicted values along the grid
npr.geyser <- function(data,tol=0.1,ftol=0.1, plot.df=evaluation.points) {
  # Find the optimal bandwidth
  bw <- npregbw(waiting ~ duration, data=data, tol=tol, ftol=ftol)
  # Now actually estimate the model
    # Two-step procedure is needed because npreg() is finicky when called inside
    # another function
  mdl <- npreg(bw)
  # Get predictions at every point in plot.df
  return(predict(mdl, newdata=plot.df))
}


## ----npr.cis, tidy=FALSE------------------------------------------------------
main.curve <- npr.geyser(geyser)

# We already defined this in a previous example, but it doesn't hurt
resample.geyser <- function() { resample.data.frame(geyser) }

geyser.resampled.curves <- rboot(statistic=npr.geyser,
                                 simulator=resample.geyser,
                                 B=800)


## ----geyser.npr.cis, dev="png", echo=FALSE, results="hide"--------------------
plot(0,type="n",xlim=c(0.8,5.5),ylim=c(0,100),
     xlab="Duration (min)", ylab="Waiting (min)")
for (i in 1:ncol(geyser.resampled.curves)) {
    lines(evaluation.points$duration,
          geyser.resampled.curves[,i], lwd=0.1, col="grey")
}
geyser.npr.cis <- bootstrap.ci(tboots=geyser.resampled.curves,
                               t.hat=main.curve, level=0.95)
lines(evaluation.points$duration, geyser.npr.cis[,"lower"])
lines(evaluation.points$duration, geyser.npr.cis[,"upper"])
lines(evaluation.points$duration, main.curve)
rug(geyser$duration,side=1)
points(geyser$duration, geyser$waiting)



## ----penn-setup---------------------------------------------------------------
penn <- read.csv("http://www.stat.cmu.edu/~cshalizi/uADA/13/hw/02/penn-select.csv")
penn.formula <- "gdp.growth ~ log(gdp) + pop.growth + invest + trade"
penn.lm <- lm(penn.formula, data=penn)




## ----penn.lm.cis--------------------------------------------------------------
# Simulate the linear model based on the Penn data by resampling residuals
# Inputs: none
# Output: new data frame where the response variable (gdp.growth) is
  # drawn from the estimated linear model plus resampled residuals
# Presumes: the data frame penn is around
  # the model penn.lm has been estimated
resample.residuals.penn <- function() {
  new.frame <- penn
  new.growths <- fitted(penn.lm) + resample(residuals(penn.lm))
  new.frame$gdp.growth <- new.growths
  return(new.frame)
}

# Estimate the Penn model on new data
# Inputs: data frame (data)
# Output: coefficient vector
# Presumes: penn.formula is defined
  # data contains columns matching the variable names in penn.formula
penn.estimator <- function(data) {
  mdl <- lm(penn.formula, data=data)
  return(coefficients(mdl))
}

penn.lm.cis <- bootstrap.ci(statistic=penn.estimator,
                            simulator=resample.residuals.penn,
                            B=1e4, t.hat=coefficients(penn.lm), level=0.95)




## -----------------------------------------------------------------------------
max.boot.ci <- function(x,B) {
    max.boot <- replicate(B,max(resample(x)))
    return(2*max(x)-quantile(max.boot,c(0.975,0.025)))
}
boot.cis <- replicate(1000,max.boot.ci(x=runif(100),B=1000))
(true.coverage <- mean((1 >= boot.cis[1,]) & (1 <= boot.cis[2,])))

