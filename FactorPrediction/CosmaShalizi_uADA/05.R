## ------------------------------------------------------------------------
6e8 * (1-0.5)^(-1/(2.33-1))

## ------------------------------------------------------------------------
6e8 * (1-0.4)^(-1/(2.33-1))

## ------------------------------------------------------------------------
1e6 * (1-0.92)^(-1/(2.5-1))

## ------------------------------------------------------------------------
# Calculate quantiles of the Pareto distribution
# Inputs: desired quantile (p)
  # exponent of the distribution (exponent)
  # lower threshold of the distribution (threshold)
# Outputs: the pth quantile
qpareto.1 <- function(p, exponent, threshold) {
  q <- threshold*((1-p)^(-1/(exponent-1)))
  return(q)
}

## ------------------------------------------------------------------------
qpareto.1(p=0.5,exponent=2.33,threshold=6e8)
qpareto.1(p=0.4,exponent=2.33,threshold=6e8)
qpareto.1(p=0.92,exponent=2.5,threshold=1e6)

## ------------------------------------------------------------------------
# Calculate quantiles of the Pareto distribution
# Inputs: desired quantile (p)
  # exponent of the distribution (exponent)
  # lower threshold of the distribution (threshold)
  # flag for whether to give lower or upper quantiles (lower.tail)
# Outputs: the pth quantile
qpareto.2 <- function(p, exponent, threshold, lower.tail=TRUE) {
  if(lower.tail==FALSE) {
    p <- 1-p
  }
  q <- threshold*((1-p)^(-1/(exponent-1)))
  return(q)
}

## ------------------------------------------------------------------------
qpareto.2(p=0.5,exponent=2.33,threshold=6e8,lower.tail=TRUE)
qpareto.2(p=0.5,exponent=2.33,threshold=6e8)
qpareto.2(p=0.92,exponent=2.5,threshold=1e6)
qpareto.2(p=0.5,exponent=2.33,threshold=6e8,lower.tail=FALSE)
qpareto.2(p=0.92,exponent=2.5,threshold=1e6,lower.tail=FALSE)

## ------------------------------------------------------------------------
# Calculate quantiles of the Pareto distribution
# Inputs: desired quantile (p)
  # exponent of the distribution (exponent)
  # lower threshold of the distribution (threshold)
  # flag for whether to give lower or upper quantiles (lower.tail)
# Outputs: the pth quantile
qpareto.3 <- function(p, exponent, threshold, lower.tail=TRUE) {
  if(lower.tail==FALSE) {
    p <- 1-p
  }
  q <- qpareto.1(p, exponent, threshold)
  return(q)
}

## ------------------------------------------------------------------------
# Calculate quantiles of the Pareto distribution
# Inputs: desired quantile (p)
  # exponent of the distribution (exponent)
  # lower threshold of the distribution (threshold)
  # flag for whether to give lower or upper quantiles (lower.tail)
# Outputs: the pth quantile
qpareto.4 <- function(p, exponent, threshold, lower.tail=TRUE) {
  stopifnot(p >= 0, p <= 1, exponent > 1, threshold > 0)
  q <- qpareto.3(p,exponent,threshold,lower.tail)
  return(q)
}

## ------------------------------------------------------------------------
qpareto.4(p=0.5,exponent=2.33,threshold=6e8,lower.tail=TRUE)
qpareto.4(p=0.92,exponent=2.5,threshold=1e6,lower.tail=FALSE)
qpareto.4(p=1.92,exponent=2.5,threshold=1e6,lower.tail=FALSE)
qpareto.4(p=-0.02,exponent=2.5,threshold=1e6,lower.tail=FALSE)
qpareto.4(p=0.92,exponent=0.5,threshold=1e6,lower.tail=FALSE)
qpareto.4(p=0.92,exponent=2.5,threshold=-1,lower.tail=FALSE)
qpareto.4(p=-0.92,exponent=2.5,threshold=-1,lower.tail=FALSE)

## ----rpareto-with-deliberate-bug-----------------------------------------
# Generate random numbers from the Pareto distribution
# Inputs: number of random draws (n)
  # exponent of the distribution (exponent)
  # lower threshold of the distribution (threshold)
# Outputs: vector of random numbers
rpareto <- function(n,exponent,threshold) {
  x <- vector(length=n)
  for (i in 1:n) {
    x[i] <- qpareto.4(p=rnorm(1),exponent=exponent,threshold=threshold)
  }
  return(x)
}

## ------------------------------------------------------------------------
rpareto(10)

## ----eval=FALSE----------------------------------------------------------
## traceback()
## ## 3: stopifnot(p >= 0, p <= 1, exponent > 1, threshold > 0) at #2
## ## 2: qpareto.4(p = rnorm(1), exponent = exponent, threshold = threshold) at #4
## ## 1: rpareto(10)

## ------------------------------------------------------------------------
rpareto(n=10,exponent=2.5,threshold=1)

## ------------------------------------------------------------------------
p = rnorm(1)

## ----rpareto-without-deliberate-bug--------------------------------------
# Generate random numbers from the Pareto distribution
# Inputs: number of random draws (n)
  # exponent of the distribution (exponent)
  # lower threshold of the distribution (threshold)
# Outputs: vector of random numbers
rpareto <- function(n,exponent,threshold) {
  x <- vector(length=n)
  for (i in 1:n) {
    x[i] <- qpareto.4(p=runif(1),exponent=exponent,threshold=threshold)
  }
  return(x)
}

## ------------------------------------------------------------------------
rpareto(n=10,exponent=2.5,threshold=1)

## ------------------------------------------------------------------------
r <- rpareto(n=1e4,exponent=2.5,threshold=1)
qpareto.4(p=0.5,exponent=2.5,threshold=1)
quantile(r,0.5)
qpareto.4(p=0.1,exponent=2.5,threshold=1)
quantile(r,0.1)
qpareto.4(p=0.9,exponent=2.5,threshold=1)
quantile(r,0.9)

## ----simulation-vs-theory-quantiles, echo=FALSE--------------------------
simulated.percentiles <- quantile(r,(0:99)/100)
theoretical.percentiles <- qpareto.4((0:99)/100,exponent=2.5,threshold=1)
plot(theoretical.percentiles,simulated.percentiles)
abline(0,1)

## ------------------------------------------------------------------------
# Compare random draws from Pareto distribution to theoretical quantiles
# Inputs: None
# Outputs: None
# Side-effects: Adds points showing random draws vs. theoretical quantiles
  # to current plot
pareto.sim.vs.theory <- function() {
  r <- rpareto(n=1e4,exponent=2.5,threshold=1)
  simulated.percentiles <- quantile(r,(0:99)/100)
  points(theoretical.percentiles,simulated.percentiles)
}

## ----simulation-vs-theory-quantiles-many, echo=FALSE---------------------
simulated.percentiles <- quantile(r,(0:99)/100)
theoretical.percentiles <- qpareto.4((0:99)/100,exponent=2.5,threshold=1)
plot(theoretical.percentiles,simulated.percentiles)
abline(0,1)
for (i in 1:10) { pareto.sim.vs.theory() }

## ------------------------------------------------------------------------
# Compare random draws from Pareto distribution to theoretical quantiles
# Inputs: Graphical arguments, passed to points()
# Outputs: None
# Side-effects: Adds points showing random draws vs. theoretical quantiles
  # to current plot
pareto.sim.vs.theory <- function(...) {
  r <- rpareto(n=1e4,exponent=2.5,threshold=1)
  simulated.percentiles <- quantile(r,(0:99)/100)
  points(theoretical.percentiles,simulated.percentiles,...)
}

## ----simulation-vs-theory-quantiles-many-2, echo=FALSE-------------------
simulated.percentiles <- quantile(r,(0:99)/100)
theoretical.percentiles <- qpareto.4((0:99)/100,exponent=2.5,threshold=1)
plot(theoretical.percentiles,simulated.percentiles)
abline(0,1)
for (i in 1:10) {
  pareto.sim.vs.theory(pch=i,type="b",lty=i)
}

## ------------------------------------------------------------------------
# Check Pareto random number generator, by repeatedly generating random draws
  # and comparing them to theoretical quantiles
# Inputs: Number of random points to generate per replicate (n)
  # exponent of distribution (exponent)
  # lower threshold of distribution (threshold)
  # number of replicates to run (B)
# Outputs: None
# Side-effects: Creates new plot, plots simulated points vs. theory
check.rpareto <- function(n=1e4,exponent=2.5,threshold=1,B=10) {
  # One set of percentiles for everything
  theoretical.percentiles <- qpareto.4((0:99)/100,exponent=exponent,
                                       threshold=threshold)
  # Set up plotting window, but don't put anything in it:
  plot(0,type="n", xlim=c(0,max(theoretical.percentiles)),
       # No more horizontal room than we need
       ylim=c(0,1.1*max(theoretical.percentiles)),
       # Allow some extra vertical room for noise
       xlab="theoretical percentiles", ylab="simulated percentiles",
       main = paste("exponent = ", exponent, ", threshold = ", threshold))
  # Diagonal, for visual reference
  abline(0,1)
  for (i in 1:B) {
    pareto.sim.vs.theory(n=n,exponent=exponent,threshold=threshold,
                            pch=i,type="b",lty=i)
  }
}

## ------------------------------------------------------------------------
# Compare random draws from Pareto distribution to theoretical quantiles
# Inputs: Graphical arguments, passed to points()
# Outputs: None
# Side-effects: Adds points showing random draws vs. theoretical quantiles
  # to current plot
pareto.sim.vs.theory <- function(n,exponent,threshold,...) {
  r <- rpareto(n=n,exponent=exponent,threshold=threshold)
  simulated.percentiles <- quantile(r,(0:99)/100)
  points(theoretical.percentiles,simulated.percentiles,...)
}

## ----theory-vs-simulation-adequate, echo=FALSE---------------------------
check.rpareto()

## ----simulation-vs-theory-bad, echo=FALSE--------------------------------
check.rpareto(n=1e4,exponent=2.33,threshold=9e8)

## ------------------------------------------------------------------------
x <- 7
x
square <- function(y) { x <- y^2; return(x) }
square(7)
x

## ------------------------------------------------------------------------
# Compare random draws from Pareto distribution to theoretical quantiles
# Inputs: Number of random points to generate (n)
  # exponent of distribution (exponent)
  # lower threshold of distribution (threshold)
  # graphical arguments, passed to points() (...)
# Outputs: None
# Side-effects: Adds points showing random draws vs. theoretical quantiles
  # to current plot
pareto.sim.vs.theory <- function(n,exponent,threshold,...) {
  r <- rpareto(n=n,exponent=exponent,threshold=threshold)
  theoretical.percentiles <- qpareto.4((0:99)/100,exponent=exponent,
                                       threshold=threshold)
  simulated.percentiles <- quantile(r,(0:99)/100)
  points(theoretical.percentiles,simulated.percentiles,...)
}

## ------------------------------------------------------------------------
# Compare random draws from Pareto distribution to theoretical quantiles
# Inputs: Graphical arguments, passed to points()
# Outputs: None
# Side-effects: Adds points showing random draws vs. theoretical quantiles
  # to current plot
check.rpareto <- function(n=1e4,exponent=2.5,threshold=1,B=10) {
  # One set of percentiles for everything
  theoretical.percentiles <- qpareto.4((0:99)/100,exponent=exponent,
    threshold=threshold)
  # Set up plotting window, but don't put anything in it:
  plot(0,type="n", xlim=c(0,max(theoretical.percentiles)),
    # No more horizontal room than we need
    ylim=c(0,1.1*max(theoretical.percentiles)),
    # Allow some extra vertical room for noise
    xlab="theoretical percentiles", ylab="simulated percentiles",
    main = paste("exponent = ", exponent, ", threshold = ", threshold))
  # Diagonal, for visual reference
  abline(0,1)
  for (i in 1:B) {
    pareto.sim.vs.theory(n=n,exponent=exponent,threshold=threshold,
      theoretical.percentiles=theoretical.percentiles,
      pch=i,type="b",lty=i)
  }
}

# Compare random draws from Pareto distribution to theoretical quantiles
# Inputs: Number of random points to generate (n)
  # exponent of distribution (exponent)
  # lower threshold of distribution (threshold)
  # vector of theoretical percentiles (theoretical.percentiles)
  # graphical arguments, passed to points()
# Outputs: None
# Side-effects: Adds points showing random draws vs. theoretical quantiles
  # to current plot
pareto.sim.vs.theory <- function(n,exponent,threshold,
  theoretical.percentiles,...) {
  r <- rpareto(n=n,exponent=exponent,threshold=threshold)
  simulated.percentiles <- quantile(r,(0:99)/100)
  points(theoretical.percentiles,simulated.percentiles,...)
}

## ----simulation-vs-theory-good, echo=FALSE-------------------------------
check.rpareto(1e4,2.33,9e8)

## ----rpareto-with-replicate----------------------------------------------
# Generate random numbers from the Pareto distribution
# Inputs: number of random draws (n)
  # exponent of the distribution (exponent)
  # lower threshold of the distribution (threshold)
# Outputs: vector of random numbers
rpareto <- function(n,exponent,threshold) {
  x <- replicate(n,qpareto.4(p=runif(1),exponent=exponent,threshold=threshold))
  return(x)
}

## ----rpareto-vectorized--------------------------------------------------
# Generate random numbers from the Pareto distribution
# Inputs: number of random draws (n)
  # exponent of the distribution (exponent)
  # lower threshold of the distribution (threshold)
# Outputs: vector of random numbers
rpareto <- function(n,exponent,threshold) {
  x <- qpareto.4(p=runif(n),exponent=exponent,threshold=threshold)
  return(x)
}

## ------------------------------------------------------------------------
# Calculate Huber's loss function
# Input: vector of numbers x
# Return: x^2 for |x|<1, 2|x|-1 otherwise
huber <- function(x) {
  n <- length(x)
  y <- vector(n)
  for (i in 1:n) {
    if (abs(x) <= 1) {
      y[i] <- x[i]^2
    } else {
      y[i] <- 2*abs(x[i])-1
    }
  }
  return(y)
}

## ------------------------------------------------------------------------
# Calculate Huber's loss function
# Input: vector of numbers x
# Return: x^2 for |x|<1, 2|x|-1 otherwise
huber <- function(x) {
  return(ifelse(abs(x) <= 1, x^2, 2*abs(x)-1))
}

## ----huber, echo=FALSE, out.width="0.5\\textwidth"-----------------------
curve(x^2,col="grey",from=-5,to=5,ylab="")
curve(huber,add=TRUE)

## ------------------------------------------------------------------------
y <- c(0.9,0.99,0.999,0.99999)
lapply(y,qpareto.4,exponent=2.5,threshold=1)

## ------------------------------------------------------------------------
sapply(y,qpareto.4,exponent=2.5,threshold=1)

## ------------------------------------------------------------------------
sapply(fiveyear.models.2, coefficients)

## ------------------------------------------------------------------------
gaussian.mle <- function(x) {
  n <- length(x)
  mean.est <- mean(x)
  var.est <- var(x)*(n-1)/n
  est <- list(mean=mean.est, sd=sqrt(var.est))
  return(est)
}

## ------------------------------------------------------------------------
x <- 1:10
mean(x)
var(x) * (length(x)-1)/length(x)
sqrt(var(x) * (length(x)-1)/length(x))
gaussian.mle(x)

## ------------------------------------------------------------------------
x <- rnorm(n=100,mean=3,sd=2)

## ------------------------------------------------------------------------
x <- rnorm(n=100,mean=3,sd=2)
median(x)

## ------------------------------------------------------------------------
medians <- vector(length=100)
for (i in 1:100) {
  x <- rnorm(n=100,mean=3,sd=2)
  medians[i] <- median(x)
}
se.in.median <- sd(medians)

## ------------------------------------------------------------------------
# Inputs: None (everything is hard-coded)
# Output: the standard error in the median
find.se.in.median <- function() {
  # Set up a vector to store the simulated medians
  medians <- vector(length=100)
  # Do the simulation 100 times
  for (i in 1:100) {
    x <- rnorm(n=100,mean=3,sd=2) # Simulate
    medians[i] <- median(x) # Calculate the median of the simulation
  }
  se.in.median <- sd(medians) # Take standard deviation
  return(se.in.median)
}

## ------------------------------------------------------------------------
# Inputs: Number of replicates (B)
# Output: the standard error in the median
find.se.in.median <- function(B) {
  # Set up a vector to store the simulated medians
  medians <- vector(length=B)
  # Do the simulation B times
  for (i in 1:B) {
    x <- rnorm(n=100,mean=3,sd=2) # Simulate
    medians[i] <- median(x) # Calculate median of the simulation
  }
  se.in.median <- sd(medians) # Take standard deviation
  return(se.in.median)
}

## ------------------------------------------------------------------------
find.se.in.median.exp <- function(B) {
  # Set up a vector to store the simulated medians
  medians <- vector(length=B)
  # Do the simulation B times
  for (i in 1:B) {
    x <- rexp(n=37,rate=2) # Simulate
    medians[i] <- median(x) # Calculate median of the simulation
  }
  se.in.median <- sd(medians) # Take standard deviation
  return(se.in.median)
}

## ------------------------------------------------------------------------
# Inputs: number of replicates (B)
  # flag for whether to use a normal or an exponential (use.norm)
# Output: The standard error in the median
find.se.in.median <- function(B,use.norm=TRUE) {
  medians <- vector(length=B)
  for (i in 1:B) {
    if (use.norm) {
      x <- rnorm(100,3,2)
    } else {
      x <- rexp(37,2)
    }
    medians[i] <- median(x)
  }
  se.in.median <- sd(medians)
  return(se.in.median)
}

## ------------------------------------------------------------------------
# Inputs: Number of replicates (B)
  # Simulator function (simulator)
# Presumes: simulator is a no-argument function which produce a vector of
  # numbers
# Output: The standard error in the media
find.se.in.median <- function(B,simulator) {
  median <- vector(length=B)
  for (i in 1:B) {
    x <- simulator()
    medians[i] <- median(x)
  }
  se.in.medians <- sd(medians)
  return(se.in.medians)
}

## ------------------------------------------------------------------------
# Inputs: None
# Output: ten draws from the mean 3, s.d. 2 Gaussian
simulator.1 <- function() {
  return(rnorm(10,3,2))
}

## ------------------------------------------------------------------------
# Inputs: None
# Output: 37 draws from the rate 2 exponential
simulator.2 <- function() {
  return(rexp(37,2))
}

## ------------------------------------------------------------------------
find.se.in.median(B=100, simulator=simulator.2)

## ------------------------------------------------------------------------
# Perturb the currency-undervaluation data by re-sampling and fit a kernel
  # regression for growht on initial GDP and undervaluation
# Inputs: None
# Output: The fitted growth rates from a new kernel regression
simulator.3 <- function() {
  # Make sure the np library is loaded
  require(np)
  # If we haven't already loaded the data, load it
  if (!exists("uv")) {
    uv <- read.csv("http://www.stat.cmu.edu/~cshalizi/uADA/16/hw/02/uv.csv")
  }
  # How big is the data set?
  n <- nrow(uv)
  # Treat the data set like a population and draw a sample
  resampled.rows <- sample(1:n,size=n,replace=TRUE)
  uv.r <- uv[resampled.rows,]
  # See the chapter on smoothing for the following incantation
  fit <- npreg(growth~log(gdp)+underval, data=uv.r, tol=1e-2, ftol=1e-2)
  growth.rates <- fitted(fit)
  return(growth.rates)
}

## ------------------------------------------------------------------------
find.se.in.median(B=10, simulator=simulator.3)

## ------------------------------------------------------------------------
# Inputs: number of replicates (B)
  # Simulator function (simulator)
# Presumes: simulator is a no-argument function which produces a vector of
  # numbers
# Outputs: Standard error in the median of the output of simulator
find.se.in.median <- function(B,simulator) {
  medians <- replicate(B, median(simulator()))
  se.in.median <- sd(medians)
  return(se.in.median)
}

## ------------------------------------------------------------------------
# Inputs: number of replicates (B)
  # Simulator function (simulator)
# Presumes: simulator is a no-argument function which produces a vector of
  # numbers
# Outputs: Interquartile range of the median of the output of simulator
find.iqr.of.median <- function(B,simulator) {
  medians <- replicate(B,median(simulator()))
  iqr.of.median <- IQR(medians)
  return(iqr.of.median)
}

## ------------------------------------------------------------------------
# Inputs: number of replicates (B)
  # Simulator function (simulator)
# Presumes: simulator is a no-argument function which produces a vector of
  # numbers
# Outputs: Standard error of the mean of the output of simulator
find.se.of.mean <- function(B,simulator) {
  means <- replicate(B,mean(simulator()))
  se.of.mean <- sd(means)
  return(se.of.mean)
}

## ------------------------------------------------------------------------
# Inputs: number of replicates (B)
  # Simulator function (simulator)
  # Estimator function (estimator)
  # Sample summarizer function (summarizer)
# Presumes: simulator is a no-argument function which produces a vector of
  # numbers
  # estimator is a function that takes a vector of numbers and produces one
  # output
  # summarizer takes a vector of outputs from estimator
# Outputs: Summary of the simulated distribution of estimates
summarize.sampling.dist.of.estimates <- function(B,simulator,estimator,
                                                 summarizer) {
  estimates <- replicate(B,estimator(simulator()))
  return(summarizer(estimates))
}

## ------------------------------------------------------------------------
bootstrap <- function(B,simulator,estimator,summarizer) {
  estimates <- replicate(B,estimator(simulator()))
  return(summarizer(estimates))
}

## ------------------------------------------------------------------------
bootstrap(B=100,simulator=simulator.1, estimator=median, summarizer=sd)

