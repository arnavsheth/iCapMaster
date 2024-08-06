## ----echo=FALSE----------------------------------------------------------
library(knitr)
opts_chunk$set(size="small", background="white", highlight=FALSE, cache=TRUE,
               autodep=TRUE, tidy=TRUE, tidy.opts=list(comment=FALSE))
opts_chunk$set(fig.path="/Users/crs/teaching/ADAfaEPoV/variance-and-weights/")
options(np.messages=FALSE)

## ----linear-response-and-quadratic-heteroskedasticity, echo=FALSE--------
plot(0, type="n", xlim=c(-5,5), ylim=c(-15,5))
abline(a=3, b=-2)
curve(1+x^2/2, add=TRUE, col="grey")

## ----set-up-running-example, include=FALSE-------------------------------
# Generate data for the running example
# X: Gaussian but with more-than-standard variance
x <- rnorm(100, 0, 3)
# Y: Linearly dependent on X but with variance that grows with X
# (heteroskedastic)
y <- 3-2*x + rnorm(100, 0, sapply(x, function(x){1+0.5*x^2}))

## ----x-y-scatterplot-with-ols-line, echo=FALSE---------------------------
# Plot the data
plot(x, y)
# Plot the true regression line
abline(a=3, b=-2, col="grey")
# Fit by ordinary least squares
fit.ols = lm(y~x)
# Plot that line
abline(fit.ols, lty="dashed")

## ----ols-residuals-vs-x, echo=FALSE--------------------------------------
par(mfrow=c(1,2))
plot(x, residuals(fit.ols))
plot(x, (residuals(fit.ols))^2)
par(mfrow=c(1,1))

## ----code:ols.heterosked-------------------------------------------------
# Generate more random samples from the same model and the same x values,
# but different y values
# Inputs: number of samples to generate
# Presumes: x exists and is defined outside this function
# Outputs: errors in linear regression estimates
ols.heterosked.example = function(n) {
  y = 3-2*x + rnorm(n, 0, sapply(x, function(x){1+0.5*x^2}))
  fit.ols = lm(y~x)
  # Return the errors
  return(fit.ols$coefficients - c(3,-2))
}

# Calculate average-case standard errors in linear regression estimates (SD of
# slope and intercept estimates)
# Inputs: number of samples per replication (n)
  # number of replications (m, defaults to 10,000)
# Calls: ols.heterosked.example
# Outputs: standard deviation of intercept and slope
ols.heterosked.error.stats = function(n, m=10000) {
  ols.errors.raw = t(replicate(m, ols.heterosked.example(n)))
  # transpose gives us a matrix with named columns
  intercept.se = sd(ols.errors.raw[,"(Intercept)"])
  slope.se = sd(ols.errors.raw[,"x"])
  return(c(intercept.se=intercept.se, slope.se=slope.se))
}

## ----x-y-scatterplot-with-wls-line, echo=FALSE, out.width="0.5\\textwidth"----
# Plot the data
plot(x, y)
# Plot the true regression line
abline(a=3, b=-2, col="grey")
# Fit by ordinary least squares
fit.ols = lm(y~x)
# Plot that line
abline(fit.ols, lty="dashed")
fit.wls = lm(y~x, weights=1/(1+0.5*x^2))
abline(fit.wls, lty="dotted")

## ------------------------------------------------------------------------
### As previous two functions, but with weighted regression

# Generate random sample from model (with fixed x), fit by weighted least
# squares
# Inputs: number of samples
# Presumes: x fixed outside function
# Outputs: errors in parameter estimates
wls.heterosked.example = function(n) {
  y = 3-2*x + rnorm(n, 0, sapply(x, function(x){1+0.5*x^2}))
  fit.wls = lm(y~x, weights=1/(1+0.5*x^2))
  # Return the errors
  return(fit.wls$coefficients - c(3,-2))
}

# Calculate standard errors in parameter estiamtes over many replications
# Inputs: number of samples per replication (n)
  # number of replications (m, defaults to 10,000)
# Calls: wls.heterosked.example
# Outputs: standard deviation of estimated intercept and slope
wls.heterosked.error.stats = function(n, m=10000) {
  wls.errors.raw = t(replicate(m, wls.heterosked.example(n)))
  # transpose gives us a matrix with named columns
  intercept.se = sd(wls.errors.raw[,"(Intercept)"])
  slope.se = sd(wls.errors.raw[,"x"])
  return(c(intercept.se=intercept.se, slope.se=slope.se))
}

## ----ols-residuals-with-smooth, echo=FALSE-------------------------------
plot(x, residuals(fit.ols)^2, ylab="squared residuals")
curve((1+x^2/2)^2, col="grey", add=TRUE)
require(np)
var1 <- npreg(residuals(fit.ols)^2 ~ x)
grid.x <- seq(from=min(x), to=max(x), length.out=300)
lines(grid.x, predict(var1, exdat=grid.x))

## ------------------------------------------------------------------------
fit.wls1 <- lm(y~x, weights=1/fitted(var1))
coefficients(fit.wls1)
var2 <- npreg(residuals(fit.wls1)^2 ~ x)

## ----fit-wls1, echo=FALSE------------------------------------------------
fit.wls1 <- lm(y~x, weights=1/fitted(var1))
par(mfrow=c(1,2))
plot(x, y)
abline(a=3, b=-2, col="grey")
abline(fit.ols, lty="dotted")
abline(fit.wls1, lty="dashed")
plot(x, (residuals(fit.ols))^2, ylab="squared residuals")
points(x, (residuals(fit.wls1))^2, pch=15)
lines(grid.x, predict(var1, exdat=grid.x))
var2 <- npreg(residuals(fit.wls1)^2 ~ x)
curve((1+x^2/2)^2, col="grey", add=TRUE)
lines(grid.x, predict(var2, exdat=grid.x), lty="dotted")
par(mfrow=c(1,1))

## ----fit.wls2------------------------------------------------------------
fit.wls2 <- lm(y~x, weights=1/fitted(var2))
coefficients(fit.wls2)
var3 <- npreg(residuals(fit.wls2)^2 ~ x)

## ----fit.wls3_and_4------------------------------------------------------
fit.wls3 <- lm(y~x, weights=1/fitted(var3))
coefficients(fit.wls3)
var4 <- npreg(residuals(fit.wls3)^2 ~ x)
fit.wls4 <- lm(y~x, weights=1/fitted(var4))
coefficients(fit.wls4)

## ------------------------------------------------------------------------
iterative.wls <- function(x, y, tol=0.01, max.iter=100) {
  iteration <- 1
  old.coefs <- NA
  regression <- lm(y~x)
  coefs <- coefficients(regression)
  while (is.na(old.coefs) ||
         ((max(coefs - old.coefs) > tol) && (iteration < max.iter))) {
    variance <- npreg(residuals(regression)^2 ~ x)
    old.coefs <- coefs
    iteration <- iteration+1
    regression <- lm(y~x, weights=1/fitted(variance))
    coefs <- coefficients(regression)
  }
  return(list(regression=regression, variance=variance, iterations=iteration))
}

## ----geyser-residuals, echo=FALSE----------------------------------------
library(MASS)
data(geyser)
geyser.ols <- lm(waiting ~ duration, data=geyser)
plot(geyser$duration, residuals(geyser.ols)^2, cex=0.5, pch=16,
  xlab="Duration (minutes)",
  ylab=expression("Squared residuals of linear model "(minutes^2)))
geyser.var <- npreg(residuals(geyser.ols)^2~geyser$duration)
duration.order <- order(geyser$duration)
lines(geyser$duration[duration.order], fitted(geyser.var)[duration.order])
abline(h=summary(geyser.ols)$sigma^2, lty="dashed")
legend("topleft",
  legend=c("data", "kernel variance", "homoskedastic (OLS)"),
  lty=c("blank", "solid", "dashed"), pch=c(16, NA, NA))

## ----include=FALSE-------------------------------------------------------
# copied from the simulation chapter, changing only the name of
# the homoskedastic model of the geyser
rgeyser <- function() {
  new.waiting <- rnorm(nrow(geyser),
                       mean=fitted(geyser.ols),
                       sd=summary(geyser.ols)$sigma)
  new.geyser <- data.frame(duration=geyser$duration, waiting=new.waiting)
  return(new.geyser)
}

## ----geyser-cond-var-with-sims, echo=FALSE-------------------------------
duration.grid <- seq(from=min(geyser$duration), to=max(geyser$duration),
                     length.out=300)
plot(duration.grid, predict(geyser.var, exdat=duration.grid),
     ylim=c(0,300), type="l",
     xlab="Duration (minutes)",
     ylab=expression("Squared residuals of linear model "(minutes^2)))
abline(h=summary(geyser.ols)$sigma^2, lty="dashed")
one.var.func <- function() {
  fit <- lm(waiting ~ duration, data=rgeyser())
  var.func <- npreg(residuals(fit)^2 ~ geyser$duration)
  lines(duration.grid, predict(var.func, exdat=duration.grid), col="grey")
}
invisible(replicate(30, one.var.func()))



## ----tricubic, echo=FALSE, out.width="0.5\\textwidth", fig.align="center"----
curve((1-abs(x)^3)^3, from=-1, to=1, ylab="tricubic function")

## ----edge-bias-from-kernel-regression, echo=FALSE------------------------
x <- runif(30, max=3)
y <- 9-x^2 + rnorm(30, sd=0.1)
plot(x, y)
rug(x, side=1, col="grey")
rug(y, side=2, col="grey")
curve(9-x^2, col="grey", add=TRUE, lwd=3)
grid.x <- seq(from=0, to=3, length.out=300)
np0 <- npreg(y~x)
lines(grid.x, predict(np0, exdat=grid.x))
np1 <- npreg(y~x, regtype="ll")
lines(grid.x, predict(np1, exdat=grid.x), lty="dashed")

