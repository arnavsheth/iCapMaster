## ----fig:gx_scatterplot, echo=FALSE--------------------------------------
x  <- runif(300,0,3)
yg <- log(x+1)+rnorm(length(x),0,0.15)
gframe <- data.frame(x=x,y=yg)
plot(x,yg,xlab="x",ylab="y",pch=16,cex=0.5)
curve(log(1+x),col="grey",add=TRUE,lwd=4)

## ----glinfit-------------------------------------------------------------
glinfit = lm(y~x,data=gframe)
print(summary(glinfit),signif.stars=FALSE,digits=2)

## ----fig:gx_with_regression_line, echo=FALSE-----------------------------
plot(x,yg,xlab="x",ylab="y",pch=16,cex=0.5)
curve(log(1+x),col="grey",add=TRUE,lwd=4)
abline(glinfit,lwd=4)

## ------------------------------------------------------------------------
signif(mean(residuals(glinfit)^2),3)

## ----first-np-fit, messages=FALSE----------------------------------------
library(np)
gnpr <- npreg(y~x,data=gframe)
signif(gnpr$MSE,3)

## ------------------------------------------------------------------------
signif((t.hat = mean(glinfit$residual^2) - gnpr$MSE),3)

## ----code:sim.lm---------------------------------------------------------
# One surrogate data set for simple linear regression
# Inputs: linear model (linfit), x values at which to
  # simulate (test.x)
# Outputs: Data frame with columns x and y
sim.lm <- function(linfit, test.x) {
  n <- length(test.x)
  sim.frame <- data.frame(x=test.x)
    # Add the y column later
  sigma <- summary(linfit)$sigma*(n-2)/n  # MLE value
  y.sim <- predict(linfit,newdata=sim.frame)
  y.sim <- y.sim + rnorm(n,0,sigma) # Add noise
  sim.frame <- data.frame(sim.frame,y=y.sim) # Adds column
  return(sim.frame)
}

## ----code:calc.T---------------------------------------------------------
# Calculate the difference-in-MSEs test statistic
# Inputs: A data frame (data)
# Output: Difference in MSEs between linear model and
  # kernel smoother
# Presumes: data has columns "x" and "y", which are input
  # and response
# Calls: np::npreg
calc.T <- function(data) {
  # Fit the linear model, extract residuals, calculate MSE
  MSE.p <- mean((lm(y~x, data=data)$residuals)^2)
  # npreg gets unhappy when called with a "data" argument
  # that is defined inside this function; npregbw does
  # not complain
  MSE.np.bw <- npregbw(y~x,data=data)
  MSE.np <- npreg(MSE.np.bw)$MSE
  return(MSE.p - MSE.np)
}

## ------------------------------------------------------------------------
calc.T(sim.lm(glinfit,x))

## ------------------------------------------------------------------------
null.samples.T <- replicate(200,calc.T(sim.lm(glinfit,x)))

## ------------------------------------------------------------------------
sum(null.samples.T > t.hat)

## ----null-dist-of-t-from-gx, echo=FALSE----------------------------------
hist(null.samples.T,n=31,xlim=c(min(null.samples.T),1.1*t.hat),probability=TRUE)
abline(v=t.hat)

## ----linear-truth, echo=FALSE--------------------------------------------
y2 <- 0.2+0.5*x + rnorm(length(x),0,0.15)
y2.frame <- data.frame(x=x,y=y2)
plot(x,y2,xlab="x",ylab="y")
abline(0.2,0.5,col="grey",lwd=2)

## ----null-dist-under-null-hyp, results="hide", echo=FALSE----------------
y2.fit <- lm(y~x,data=y2.frame)
null.samples.T.y2 <- replicate(200,calc.T(sim.lm(y2.fit,x)))
t.hat2 <- calc.T(y2.frame)
hist(null.samples.T.y2,n=31,probability=TRUE)
abline(v=t.hat2)

## ----hx_curve, echo=FALSE------------------------------------------------
h <- function(x) { 0.2 + 0.5*(1+sin(x)/10)*x }
curve(h(x),from=0,to=3)

## ----nearly.linear-------------------------------------------------------
nearly.linear.out.of.sample = function(n) {
  # Combines simulating the true model with fitting
  # parametric model and smoother, calculating MSEs
  x <- seq(from=0,to=3,length.out=n)
  y <- h(x) + rnorm(n,0,0.15)
  data <- data.frame(x=x,y=y)
  y.new <- h(x) + rnorm(n,0,0.15)
  sim.lm <- lm(y~x,data=data)
  lm.mse <- mean(( fitted(sim.lm) - y.new )^2)
  sim.np.bw <- npregbw(y~x,data=data)
  sim.np <- npreg(sim.np.bw)
  np.mse <- mean((fitted(sim.np) - y.new)^2)
  mses <- c(lm.mse,np.mse)
  return(mses)
}

nearly.linear.generalization <- function(n,m=100) {
  raw <- replicate(m,nearly.linear.out.of.sample(n))
  reduced <- rowMeans(raw)
  return(reduced)
}

## ----swapping-generalization-errors, results="hide", echo=FALSE----------
sizes <- c(5,10,15,20,25,30,50,100,200,500,1000)
generalizations <- sapply(sizes,nearly.linear.generalization)
plot(sizes,sqrt(generalizations[1,]), type="l",
     xlab="n",ylab="RMS generalization error", log="xy",
     ylim=c(0.14, 0.3))
lines(sizes,sqrt(generalizations[2,]),lty="dashed")
abline(h=0.15,col="grey")

