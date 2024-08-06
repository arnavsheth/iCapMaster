## ----include=FALSE------------------------------------------------------------
##### _Advanced Data Analysis from an Elementary Point of View_ #####
# R code for the chapter "Splines"
# Please do not re-distribute or use without attribution
# http://www.stat.cmu.edu/~cshalizi/ADAfaEPoV/






## ----s-and-p-stock-example----------------------------------------------------
require(pdfetch)
sp <- pdfetch_YAHOO("SPY", fields="adjclose",
  from=as.Date("1993-02-09"), to=as.Date("2015-02-09"))
sp <- diff(log(sp))
# need to drop the initial NA which makes difficulties later
sp <- sp[-1]


## ----our-first-spline, warning=FALSE------------------------------------------
sp.today <- head(sp,-1)
sp.tomorrow <- tail(sp,-1)
coefficients(lm(sp.tomorrow ~ sp.today))
sp.spline <- smooth.spline(x=sp.today,y=sp.tomorrow,cv=TRUE)
sp.spline
sp.spline$lambda


## ----some-spline-fits, echo=FALSE, warning=FALSE------------------------------
plot(as.vector(sp.today),as.vector(sp.tomorrow),xlab="Today's log-return",
     ylab="Tomorrow's log-return",pch=16,cex=0.5,col="grey")
abline(lm(sp.tomorrow ~ sp.today),col="darkgrey")
sp.spline <- smooth.spline(x=sp.today,y=sp.tomorrow,cv=TRUE)
lines(sp.spline)
lines(smooth.spline(sp.today,sp.tomorrow,spar=1.5),col="blue")
lines(smooth.spline(sp.today,sp.tomorrow,spar=2),col="blue",lty=2)
lines(smooth.spline(sp.today,sp.tomorrow,spar=1.1),col="red")
lines(smooth.spline(sp.today,sp.tomorrow,spar=0.5),col="red",lty=2)



## ----first-example-of-predict.smooth.spline-----------------------------------
predict(sp.spline,x=0.01)


## ----resampling-the-data-frame------------------------------------------------
# Bundle the two variables into a data frame
sp.frame <- data.frame(today=sp.today,tomorrow=sp.tomorrow)

# Resample rows from the S&P data frame
# Inputs: none
# Output: new data frame of same size as real data
# Presumes: sp.frame exists (and is a data frame)
# Exercise: Re-write the bootstrap demo which follows using the functions
  # from the bootstrap chapter
sp.resampler <- function() {
  n <- nrow(sp.frame)
  resample.rows <- sample(1:n,size=n,replace=TRUE)
  return(sp.frame[resample.rows,])
}


## ----sp.spline.estimator------------------------------------------------------
# Set up a grid of evenly-spaced points on which to evaluate the spline
grid.300 <- seq(from=min(sp.today),to=max(sp.today),length.out=300)

# Estimate a spline from data and return its predictions on a fixed grid
# Inputs: data frame (data)
  # one-dimensional vector of points at which to evaluate spline (eval.grid)
# Output: vector of predictions
# Presumes: first two columns of data contain predictor and response variables
sp.spline.estimator <- function(data, eval.grid=grid.300) {
  # Fit spline to data, with cross-validation to pick lambda
  fit <- smooth.spline(x=data[,1],y=data[,2],cv=TRUE)
  # Do the prediction on the grid and return the predicted values
  return(predict(fit,x=eval.grid)$y)  # We only want the predicted values
}


## ----sp.spline.cis------------------------------------------------------------
# Find confidence bands for a spline by row resampling
# Inputs: number of bootstrap replicates (B)
  # error probability / 1-confidence level (alpha)
  # grid of points on which to evaluate spline (eval.grid)
# Output: list containing a vector giving values of the spline curve
  # fit to all data along the grid (main.curve), vector of lower
  # limits (lower.ci), vector of upper limits (upper.ci), the actual grid of
  # points used for evaluation (x)
# Presumes: sp.frame exists and has the right values
sp.spline.cis <- function(B, alpha, eval.grid=grid.300) {
  spline.main <- sp.spline.estimator(sp.frame, eval.grid=eval.grid)
  # Draw B boottrap samples, fit the spline to each
    # Result has length(eval.grid) rows and B columns
  spline.boots <- replicate(B,
    sp.spline.estimator(sp.resampler(), eval.grid=eval.grid))
  # See the bootstrap chapter for the following centering trick, which
    # improves the accuracy of bootstrap confidence limits
  cis.lower <- 2*spline.main - apply(spline.boots, 1, quantile, probs=1-alpha/2)
  cis.upper <- 2*spline.main - apply(spline.boots, 1, quantile, probs=alpha/2)
  # Bundle everything up and return
  return(list(main.curve=spline.main,lower.ci=cis.lower,upper.ci=cis.upper,
    x=eval.grid))
}


## ----spline-cis, echo=FALSE, warning=FALSE------------------------------------
sp.cis <- sp.spline.cis(B=1000,alpha=0.05)
plot(as.vector(sp.today),as.vector(sp.tomorrow),xlab="Today's log-return",
  ylab="Tomorrow's log-return",pch=16,cex=0.5,col="grey")
abline(lm(sp.tomorrow ~ sp.today),col="darkgrey")
lines(x=sp.cis$x,y=sp.cis$main.curve,lwd=2)
lines(x=sp.cis$x,y=sp.cis$lower.ci)
lines(x=sp.cis$x,y=sp.cis$upper.ci)

