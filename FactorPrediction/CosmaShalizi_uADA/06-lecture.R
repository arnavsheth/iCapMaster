# In-class demos for simulation, 28 January 2016

# Load this week's homework data
stocks <- read.csv("http://www.stat.cmu.edu/~cshalizi/uADA/16/hw/03/stock_history.csv")

# Look at the cumulative returns
plot(Return_cumul ~ Date, data=stocks,xlab="date",
  ylab="Cumulative value of initial $1",type="l")
plot(Return_cumul ~ Date, data=stocks,xlab="date",
  ylab="Cumulative value of initial $1",type="l",log="y")

# Create a time series of logarithmic returns
stocks$logreturns <- c(diff(log(stocks$Return_cumul)),NA)
plot(logreturns ~ Date, data=stocks,xlab="date",
  ylab="Monthly logarithmic returns",type="l")

# Look at the histogram
hist(stocks$logreturns, freq=FALSE, xlab="log returns",n=51,
  main="Distribution of log returns")

# The conventional model: log returns are IID Gaussian
  # Maximum-likelihood estimates of mean and variance are the sample values
    # Need na.rm here because of the final NA in the vector of returns
(mean.logreturns <- mean(stocks$logreturns,na.rm=TRUE))
(sd.logreturns <- sd(stocks$logreturns,na.rm=TRUE))
curve(dnorm(x,mean=mean.logreturns,sd=sd.logreturns),add=TRUE,
  lwd=3)

# How bad are the worst 1% of months, according to the model?
  # "99% value at risk" is (roughly) this fractional loss * size of portfolio
# Analytical answer:
qnorm(0.01,mean.logreturns,sd.logreturns)
  # Presumes we can work out the quantile function exactly
# Simulation answer:
  # - Create a very long simulation run (a million points)
  # - Look at the sample quantile in that long run?
quantile(rnorm(1e6,mean.logreturns,sd.logreturns),0.01)
# How bad are the worst 1% of two months in a row?
### Exercise: find the analytical answer in the Gaussian model
# Simulation answer:
  # - Make another (or the same!) long run
  # - Take sum of consecutive pairs
  # - Take quantile of the sums
norm.sim.run <- rnorm(1e6,mean.logreturns,sd.logreturns)
quantile(head(norm.sim.run,-1) + tail(norm.sim.run,-1),0.01)

# Does the model look like the real data?
# Start by plotting the real data again
plot(logreturns ~ Date, data=stocks,xlab="date",
  ylab="Monthly logarithmic returns",type="l",lwd=2)
# Now do a simulation run, matched in length to the data, and add it to the
# plot, but fainter
lines(stocks$Date, rnorm(n=nrow(stocks),mean=mean.logreturns,sd=sd.logreturns),
  col="darkgrey",lwd=1)
# This shouldn't line up perfectly (why not?), but:
  # - data has more comparatively large movements than the model
  # - spikes seem to cluster more in the data

# How bad were the worst 1% of months?
(q0.01 <- quantile(stocks$logreturns, 0.01, na.rm=TRUE))
# What's the probability that the sample 1%, over this length of time, would
# be at least that low?
  # Analytical answer: just try to find it!
  # Simulation answer: just try it out
sim.history.std <- function() {
    rnorm(n=nrow(stocks), mean=mean.logreturns, sd=sd.logreturns)
}
simd.1stpercent.1k <- replicate(1000, quantile(sim.history.std(), 0.01))
mean(simd.1stpercent.1k <= q0.01 )




# Let's try something with heavier tails: the t distribution
library(MASS)
  # The fitdistr() function fits parametric distributions by maximum
  # likelihood; it knows about the t distribution
  # R trick: wrapping an assignment in parentheses, (a<-b), does the
  # assignment and prints out the value assigned
(t.est <- fitdistr(na.omit(stocks$logreturns),"t"))
  # The standard t distribution is centered around 0 and has scale 1
  # fitdistr shifts the center by m and changes the scale by s
  # so (X-m)/s has a standard t distribution
# restore the plot with the histogram
hist(stocks$logreturns, freq=FALSE, xlab="log returns",n=51,
  main="Distribution of log returns",ylim=c(0,15))
# and the Gaussian density curve
curve(dnorm(x,mean=mean.logreturns,sd=sd.logreturns),add=TRUE,
  lwd=3)
# Add the t-distribution density curve
  # R has a built-in density for the t distribution, dt(), but we need
  # to deal with the shift and the scale
dt.fitted <- function(x,fitted.t=t.est) {
  m <- fitted.t$estimate["m"]
  s <- fitted.t$estimate["s"]
  df <- fitted.t$estimate["df"]
  return((1/s)*dt((x-m)/s,df=df)) # what the (1/s) factor?
}
# Finally, plot the density of the fitted t distribution
curve(dt.fitted(x),add=TRUE,col="blue",lwd=3)
  # Looks definitely more promising, if not necessarily right

# Write something to simulate IID draws from the fitted t distributed
  # Uses the built-in rt(), but handles the shift and the scale
rt.fitted <- function(n,fitted.t=t.est) {
  m <- fitted.t$estimate["m"]
  s <- fitted.t$estimate["s"]
  df <- fitted.t$estimate["df"]
  t <- rt(n,df=df)
  x <- s*t + m
  return(x)
}

# How does a simulated path look?
plot(logreturns ~ Date, data=stocks,xlab="date",
  ylab="Monthly logarithmic returns",type="l",lwd=2)
lines(stocks$Date, rt.fitted(n=nrow(stocks)),col="darkgrey",lwd=1)
# Much better spikiness, but still not much clustering

# How bad does this say the worst 1% of months are?
quantile(rt.fitted(1e6),0.01)

#### Maybe the IID part is wrong?  ####

# Tack 1: trends over time ("fixed effects", as in HW 2)
library(np)
# Fit a regression of returns on the date (a numerical variable)
time.trend <- npreg(logreturns ~ Date, data=stocks,tol=1e-3,ftol=1e-3)
# Restore the plot of the time series
plot(logreturns ~ Date, data=stocks,xlab="date",
  ylab="Monthly logarithmic returns",type="l",lwd=2)
# Plot the time trend
  # Not quite flat, if you zoom in
lines(head(stocks$Date,-1), fitted(time.trend), col="darkgrey",lwd=4)
# Simulate some scatter around the time trend.
  # Note: rnorm now returns a vector!
lines(head(stocks$Date,-1), rnorm(n=length(fitted(time.trend)),
                                  mean=fitted(time.trend),
				  sd=sqrt(time.trend$MSE)),
  col="blue",lwd=1)
# Looks a bit better but still isn't clustered enough
  # Exercise: could you replace rnorm() above with a t distribution?

# OK, what about dependence from one month to the next, rather than a fixed
# effect of time?
# Plot this month's returns against next
plot(head(stocks$logreturns,-1),tail(stocks$logreturns,-1),
  xlab="This month's returns",ylab="Next month's returns")
# The heads and tails are getting old; make a data frame and be done
returns.df <- data.frame(Date=head(stocks$Date,-1),
  r0=head(stocks$logreturns,-1),r1=tail(stocks$logreturns,-1))
returns.df <- head(returns.df,-1) # Remove an NA row
# Regress next month's returns on this month's
time.regression <- npreg(r1~r0,data=returns.df,tol=1e-3,ftol=1e-3)
# Create a sequence of evenly spaced values on the horizontal axis for plotting
plotting.r0 <- seq(from=-0.3,to=0.3,length.out=100)
# Add the fitted values to the plot
lines(plotting.r0,predict(time.regression,newdata=data.frame(r0=plotting.r0)),
  col="red",lwd=2)
  # Weirdness out where there are just a few data points, but a pretty
  # nearly linear trend through the bulk of the data
abline(a=0,b=1,col="green",lwd=2) # For comparison of slope
# Now, how do we use this to simulate step by step?

# Start with an empty vector to store results
  # We could keep extending a vector, but that's very slow
fake.returns <- vector(length=nrow(stocks))
# Give it an initial value for the first month
fake.returns[1] <- stocks$logreturns[1]
# Now for every new month
for (t in 2:nrow(stocks)) {
  # Work out the expected value from the previous month's simulated
    # returns
  expected <- predict(time.regression,
                      newdata=data.frame(r0=fake.returns[t-1]))
  # Add some noise and stick it in the vector
  fake.returns[t] <- expected+
    rnorm(n=1,mean=0,sd=sqrt(time.regression$MSE))
}

plot(logreturns ~ Date, data=stocks,xlab="date",
     ylab="Monthly logarithmic returns",type="l",lwd=2)
lines(stocks$Date,fake.returns,col="purple",lwd=2)
quantile(fake.returns,0.01)
  # Looks rather more convincingly clustered, but not spiky enough
  # Exercise: turn the simulation into a function, to make it eaiser to
    # re-run
  # Exercise: Replace rnorm() in the code with drawing from a t distribution

