## ----include=FALSE------------------------------------------------------------
##### _Advanced Data Analysis from an Elementary Point of View_ #####
# R code for the chapter "Smoothing"
# Please do not re-distribute or use without attribution
# http://www.stat.cmu.edu/~cshalizi/ADAfaEPoV/




## ----components-of-generalization-error, echo=FALSE, out.width="0.5\\textwidth"----
curve(2*x^4,from=0,to=1,lty=2,xlab="Smoothing",ylab="Generalization error")
curve(0.12+x-x,lty=3,add=TRUE)
curve(1/(10*x),lty=4,add=TRUE)
curve(0.12+2*x^4+1/(10*x),add=TRUE)



## ----error-components-with-more-data, out.width="0.5\\textwidth", echo=FALSE----
curve(2*x^4,from=0,to=1,lty=2,xlab="Smoothing",ylab="Generalization error")
curve(0.12+x-x,lty=3,add=TRUE)
curve(1/(10*x),lty=4,add=TRUE,col="grey")
curve(0.12+2*x^2+1/(10*x),add=TRUE,col="grey")
curve(1/(30*x),lty=4,add=TRUE)
curve(0.12+2*x^4+1/(30*x),add=TRUE)



## ----two-curves, echo=FALSE---------------------------------------------------
par(mfcol=c(2,1))
# Fix two functions for the examples
  # Full function comments skipped here
true.r <- function(x) {sin(x)*cos(20*x)}
true.s <- function(x) {log(x+1)}
# Plot them (over the same range)
curve(true.r(x), from=0, to=3, xlab="x", ylab=expression(r(x)))
curve(true.s(x), from=0, to=3, xlab="x", ylab=expression(s(x)))
par(mfcol=c(1,1))



## ----two-curves-noisy, echo=FALSE---------------------------------------------
# Scatter some x points over the range of the previous plot
x = runif(300,0,3)
# Make y = function at x plus noise
yr = true.r(x) + rnorm(length(x),0,0.15)
ys = true.s(x) + rnorm(length(x),0,0.15)
par(mfcol=c(2,1))
# Plot the noisy data plus the true curves
plot(x, yr, xlab="x", ylab=expression(r(x)+epsilon))
curve(true.r(x), col="grey", add=TRUE)
plot(x, ys, xlab="x", ylab=expression(s(x)+eta))
curve(true.s(x), col="grey", add=TRUE)



## ----local-averaging, echo=FALSE----------------------------------------------
par(mfcol=c(2,1))
# Re-do previous plots, but with everything outside a window on our favorite,
# focal value of x "ghosted out"
x.focus <- 1.6; x.lo <- x.focus-0.1; x.hi <- x.focus+0.1
# Black near focal point, else grey
colors=ifelse((x<x.hi)&(x>x.lo),"black","grey")
plot(x,yr,xlab="x",ylab=expression(r(x)+epsilon),col=colors)
curve(true.r(x),col="grey",add=TRUE)
points(x.focus,mean(yr[(x<x.hi)&(x>x.lo)]),pch=18,cex=2)
plot(x,ys,xlab="x",ylab=expression(s(x)+eta),col=colors)
curve(true.s(x),col="grey",add=TRUE)
points(x.focus,mean(ys[(x<x.hi)&(x>x.lo)]),pch=18,cex=2)
par(mfcol=c(1,1))



## ----local-averaging-errors, out.width="0.4\\textwidth", echo=FALSE-----------
# Define a little function to calculate the error of local averaging around
# a focal point, as a function of the width of the averaging window
  # Of course, this only works because we made up the true curves...
loc_ave_err <- function(h,y,y0) {abs(y0-mean(y[(x.focus-h < x) & (x.focus+h>x)]))}
yr0=true.r(x.focus); ys0=true.s(x.focus)
# Apply the averaging-error function over a range of window sizes
r.LAE = sapply(1:100/100,loc_ave_err,y=yr,y0=yr0)
s.LAE = sapply(1:100/100,loc_ave_err,y=ys,y0=ys0)
# Plot error vs. window size for the two data sets
plot(1:100/100,r.LAE,xlab="Radius of averaging window",ylim=c(0,1.1),
     ylab="Absolute value of error",type="l",log="x")
lines(1:100/100,s.LAE,lty="dashed")
abline(h=0.15,col="grey")



## -----------------------------------------------------------------------------
# Demo of bandwidth selection by cross-validation for 1-D kernel smoothing
  # ATTN: This is JUST a demo of the concept
    # It's slow, it's inflexible, and it only handles 1D
    # In general, use the built-in bandwidth selector from the np package
  # YOU SHOULD NOT USE THIS IN HOMEWORK (unless specifically told to)
  # YOU SHOULD NOT USE THIS IN REAL PROJECTS
# Inputs: vector of regressor values (x)
  # vector of regressand values (y)
  # vector of bandwidths (bandwidths; default values pretty arbitrary)
  # number of folds of cross-validation (nfolds)
# Output: list with components "bwest.bw" (number, best bandwidth),
  # "CV_MSEs" (numeric vector, CV scores of all bandwidths),
  # "fold_MSEs" (numeric matrix, score of each bandwidth on each fold)
# Presumes: np package is installed
cv_bws_npreg <- function(x,y,bandwidths=(1:50)/50,nfolds=10) {
  # Load the np package (if it's not loaded already)
  require(np)
  # How many data points?
  n <- length(x)
  # Sanity checks: more than one observation, equal number of x and y values...
  stopifnot(n > 1, length(y) == n)
  # ... more than one bandwidth...
  stopifnot(length(bandwidths) > 1)
  # ... at least one fold, fold number is an integer
  stopifnot(nfolds > 0, nfolds==trunc(nfolds))

  # Prepare a matrix to store fold-by-bandwidth out-of-sample errors
  fold_MSEs <- matrix(0,nrow=nfolds,ncol=length(bandwidths))
  # Number the columns by the bandwidths
  colnames(fold_MSEs) = bandwidths

  # Start by assigning each row to a fold, cycling through them
    # (e.g., as 1, 2, 3, 1, 2, 3, 1, 2, 3, ...) if nfolds==3
  # Then randomly permute the ordering
  case.folds <- sample(rep(1:nfolds,length.out=n))
  # Thus all folds have an equal number of random data points
  # Cycle through the folds
  for (fold in 1:nfolds) {
    # Everything not in the current fold is in the training set
    train.rows = which(case.folds!=fold)
    # Subset the data into training and testing sets
    x.train = x[train.rows]
    y.train = y[train.rows]
    x.test = x[-train.rows]
    y.test = y[-train.rows]
    # For each bandwidth,
    for (bw in bandwidths) {
      # Fit the model on the training set, and evaluate it on the testing test
      fit <- npreg(txdat=x.train,tydat=y.train,
                   exdat=x.test,eydat=y.test,bws=bw)
      # The $MSE component of the fit is calculated on the evaluation set, if
      # it's given one, so we don't need to explicitly call predict() and
      # calculate diffrences here.
      # Exercise: If npreg didn't have this feature, how would you use predict?
      fold_MSEs[fold,paste(bw)] <- fit$MSE
    }
  }
  # Take the average MSE for each bandwidth, across folds
  CV_MSEs = colMeans(fold_MSEs)
  # The best bandwidth is the one with the smallest CV'd error
  best.bw = bandwidths[which.min(CV_MSEs)]
  return(list(best.bw=best.bw,CV_MSEs=CV_MSEs,fold_MSEs=fold_MSEs))
}


## ----RMS-error-vs-bandwiths, out.width="0.5\\textwidth", echo=FALSE, message=FALSE----
# Select a bandwidth for the two data sets by 10-fold CV, over a somewhat
# arbitrary set of bandwidths
rbws <- cv_bws_npreg(x, yr, bandwidths=(1:100)/200)
sbws <- cv_bws_npreg(x, ys, bandwidths=(1:100)/200)
# Plot the square root of the cross-validated MSE, as a function of bandwidths
  # using the square root makes the error comparable in scale to the original
  # measurements
plot(1:100/200, sqrt(rbws$CV_MSEs), xlab="Bandwidth",
  ylab="Root CV MSE", type="l", ylim=c(0,0.6), log="x")
lines(1:100/200, sqrt(sbws$CV_MSEs) ,lty="dashed")
abline(h=0.15, col="grey")



## ----two-curves-plus-smooths, echo=FALSE, message=FALSE-----------------------
# What permutation of the x's would make them increasing?
  # Keep track of this to simplify plotting later
x.ord=order(x)
par(mfcol=c(2,1))
# Plot the data (as now usual)
plot(x,yr,xlab="x",ylab=expression(r(x)+epsilon))
# Get an estimate of the curve, at the best of our bandwidths
rhat <- npreg(bws=rbws$best.bw,txdat=x,tydat=yr)
# Plot the fitted values
  # We use the order of the x's here, so as not to get crazy zig-zags
lines(x[x.ord],fitted(rhat)[x.ord],lwd=4)
# Add on the true curve in grey
curve(true.r(x),col="grey",add=TRUE,lwd=2)
# The other running example gets the same treatment
plot(x,ys,xlab="x",ylab=expression(s(x)+eta))
shat <- npreg(bws=sbws$best.bw,txdat=x,tydat=ys)
lines(x[x.ord],fitted(shat)[x.ord],lwd=4)
curve(true.s(x),col="grey",add=TRUE,lwd=2)
par(mfcol=c(1,1))




## ----demo-surface, echo=FALSE, out.width="0.5\\textwidth"---------------------
# Pick evenly-spaced points for plotting on the two axes
x1.points <- seq(-3, 3, length.out=100)
x2.points <- x1.points
# Create a grid which does all possible combinations of th two coordinates
x12grid <- expand.grid(x1=x1.points, x2=x2.points)
y <- matrix(0, nrow=100, ncol=100)
# Evaluate the mystery function f at all combinations of the two coordinates
  # f is defined in a deliberately-hidden code chunk
y <- outer(x1.points, x2.points, f)
# Load the "lattice" graphics library
library(lattice)
# Do a 3D wireframe plot of y values vs. the two x coordinates
wireframe(y~x12grid$x1*x12grid$x2, scales=list(arrows=FALSE),
  xlab=expression(x^1), ylab=expression(x^2), zlab="y")



## ----demo-surface-plus-noise, echo=FALSE, out.width="0.5\\textwidth"----------
# Pick random coordinates
x1.noise <- runif(1000, min=-3, max=3)
x2.noise <- runif(1000, min=-3, max=3)
# y = mystery f at those coordinates plus noise
y.noise <- f(x1.noise, x2.noise) + rnorm(1000,0,0.05)
noise <- data.frame(y=y.noise, x1=x1.noise, x2=x2.noise)
cloud(y~x1*x2,data=noise, col="black", scales=list(arrows=FALSE),
      xlab=expression(x^1), ylab=expression(x^2), zlab="y")



## ----demo-surface-reconstruction, out.width="0.5\\textwidth", echo=FALSE------
# Use kernel regression on the noisy data
noise.np <- npreg(y~x1+x2, data=noise)
y.out <- matrix(0,100,100)
# Generate predictions on our old plotting grid
  # As usual, the fact that non of the training points were on the grid
  # coordinates doesn't matter for predict()
y.out <- predict(noise.np, newdata=x12grid)
# Plot the estimated function
wireframe(y.out~x12grid$x1*x12grid$x2, scales=list(arrows=FALSE),
          xlab=expression(x^1), ylab=expression(x^2), zlab="y")



## ----cross-section, echo=FALSE, out.width="0.5\\textwidth"--------------------
# Make a new data frame where x1 varies but x2 is pinned in place
new.frame <- data.frame(x1=seq(-3,3,length.out=300), x2=median(x2.noise))
# Plot predictions on this new frame
plot(new.frame$x1,predict(noise.np, newdata=new.frame),
  type="l",xlab=expression(x^1),ylab="y",ylim=c(0,1.0))
# Re-do with x2 decreased
new.frame$x2 <- quantile(x2.noise,0.25)
lines(new.frame$x1, predict(noise.np,newdata=new.frame), lty=2)
# Re-do with x2 increased
new.frame$x2 <- quantile(x2.noise,0.75)
lines(new.frame$x1,predict(noise.np,newdata=new.frame),lty=3)



## ----logistic-curve, out.width="0.5\\textwidth", echo=FALSE-------------------
curve(exp(7*x)/(1+exp(7*x)),from=-5,to=5,ylab="y")



## -----------------------------------------------------------------------------
# Create a random data frame, to illustrate how npreg works
# Input: number of desired rows (n)
# Outputs: data frame with n rows and 4 columns
make.demo.df <- function(n) {
  # Our desired regression function has three arguments, two quantitative
    # and one categorical
  # Inputs: numeric vector (x), numeric vector (z), character vector (w)
  # Output: numeric vector, the value of the regression function
  demo.func <- function(x,z,w) {
    # the function is linear in z if w=="A", otherwise logistic in z
    20*x^2 + ifelse(w=="A", z, 10*exp(z)/(1+exp(z)))
  }
  # Make up random x, z, w values
  x <- runif(n,-1,1)
  z <- rnorm(n,0,10)
  w <- sample(c("A","B"),size=n,replace=TRUE)
  # y is demo.func plus noise (of fixed magnitude)
  y <- demo.func(x,z,w) + rnorm(n,0,0.05)
  # Return the desired data frame
  return(data.frame(x=x,y=y,z=z,w=w))
}

demo.df <- make.demo.df(100)


## -----------------------------------------------------------------------------
demo.np1 <- npreg(y ~ x + z, data=demo.df)


## -----------------------------------------------------------------------------
summary(demo.np1)


## -----------------------------------------------------------------------------
demo.np1$MSE


## -----------------------------------------------------------------------------
demo.np1$bws$fval


## -----------------------------------------------------------------------------
predict(demo.np1, newdata=data.frame(x=-1,z=5))


## -----------------------------------------------------------------------------
demo.np3 <- npreg(y~x+z+factor(w),data=demo.df)


## -----------------------------------------------------------------------------
demo.np3$bws$fval


## ----npreg-demo-plot-1, out.width="0.5\\textwidth", echo=FALSE----------------
plot(demo.np1,theta=40,view="fixed")



## ----npreg-demo-plot-3, echo=FALSE--------------------------------------------
plot(demo.np3)



## -----------------------------------------------------------------------------
bigdemo.df <- make.demo.df(1e3)
system.time(demo.np4 <- npreg(y~x+z+factor(w), data=bigdemo.df,
                              tol=0.01,ftol=0.01))


## -----------------------------------------------------------------------------
demo.np4$bws


## ----npreg-bigdemo-plot, echo=FALSE-------------------------------------------
x.seq <- seq(from=-1,to=1,length.out=50)
z.seq <- seq(from=-30,to=30,length.out=50)
grid.A <- expand.grid(x=x.seq,z=z.seq,w="A")
grid.B <- expand.grid(x=x.seq,z=z.seq,w="B")
yhat.A <- predict(demo.np4,newdata=grid.A)
yhat.B <- predict(demo.np4,newdata=grid.B)
par(mfrow=c(1,2))
persp(x=x.seq,y=z.seq,z=matrix(yhat.A,nrow=50),theta=40,main="W=A",
  xlab="x",ylab="z",zlab="y",ticktype="detailed")
persp(x=x.seq,y=z.seq,z=matrix(yhat.B,nrow=50),theta=40,main="W=B",
  xlab="x",ylab="z",zlab="y",ticktype="detailed")

