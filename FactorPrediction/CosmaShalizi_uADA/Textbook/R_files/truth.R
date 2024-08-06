## ----include=FALSE------------------------------------------------------------
##### _Advanced Data Analysis from an Elementary Point of View_ #####
# R code for the chapter "The Truth about Linear Regression"
# Please do not re-distribute or use without attribution
# http://www.stat.cmu.edu/~cshalizi/ADAfaEPoV/




## ----slope-varies-with-distribution, echo=FALSE-------------------------------
# Create three distributions for X
x1 <- runif(100)
x2 <- rnorm(100,0.5,0.1)
x3 <- runif(100,2,3)
# Create matching Y variables from the same (nonlinear) model
y1 <- sqrt(x1) + rnorm(length(x1),0,0.05)
y2 <- sqrt(x2) + rnorm(length(x2),0,0.05)
y3 <- sqrt(x3) + rnorm(length(x3),0,0.05)
# Plot the first set of (X,Y) points, making sure the plotting region is big
  # enough for all the later ones
plot(x1,y1,xlim=c(0,3),ylim=c(0,3), xlab="X", ylab="Y", col="darkgreen", pch=15)
# Rugs for the those points, to indicate the marginal distribution
rug(x1,side=1, col="darkgreen")
rug(y1,side=2, col="darkgreen")
# Add the second set of points in a different color and plotting symbol
points(x2,y2,pch=16,col="blue")
rug(x2,side=1,col="blue")
rug(y2,side=2,col="blue")
# And the third
points(x3,y3,pch=17,col="red")
rug(x3,side=1,col="red")
rug(y3,side=2,col="red")
# Fit the regression lines and add them, in matching colors, with different
  # line styles
lm1 <- lm(y1 ~ x1)
lm2 <- lm(y2 ~ x2)
lm3 <- lm(y3 ~ x3)
abline(lm1, col="darkgreen", lty="dotted")
abline(lm2, col="blue", lty="dashed")
abline(lm3, col="red", lty="dotdash")
# Combine the data, fit an over-all regression line
x.all<-c(x1,x2,x3)
y.all<-c(y1,y2,y3)
lm.all <- lm(y.all~x.all)
abline(lm.all,lty="solid")
# Finally, the true regression curve.
curve(sqrt(x),col="grey",add=TRUE)
legend("topleft", legend=c("Unif[0,1]", "N(0.5, 0.01)", "Unif[2,3]",
                           "Union of above", "True regression line"),
       col=c("black", "blue", "red", "black", "grey"), pch=c(15,16,17, NA, NA),
       lty=c("dotted", "dashed", "dotdash", "solid", "solid"))




## ----scatterplot-for-omitted-variables, echo=FALSE, out.width="0.5\\textwidth"----
# Make the 3D plot to show omitted variable bias
library(lattice)
library(MASS)  # for multivariate normal generator

# Make correlated normal variables X and Z
x.z = mvrnorm(100, c(0,0), matrix(c(1,0.1,0.1,1), nrow=2))
# Y = X+Z + small noise
y = x.z[,1] + x.z[,2] + rnorm(100, 0, 0.1)
# 3D scatterplot, with tick-marks on axes rather than just arrows
cloud(y~x.z[,1]*x.z[,2], xlab="X", ylab="Z", zlab="Y",
      scales=list(arrows=FALSE), col.point="black")



## ----scatterplot-for-omitted-variables-post-shift, echo=FALSE, out.width="0.5\\textwidth"----
# Continuation of previous example
# Change the correlation between X and Z to -0.1 instead of +0.1
new.x.z = mvrnorm(100,c(0,0),matrix(c(1,-0.1,-0.1,1),nrow=2))
new.y = new.x.z[,1] + new.x.z[,2] + rnorm(100,0,0.1)
cloud(new.y~new.x.z[,1]*new.x.z[,2], xlab="X", ylab="Z", zlab="Y",
      scales=list(arrows=FALSE))



## ----y-on-x-with-z-shifted, echo=FALSE, out.width="0.5\\textwidth"------------
# Continuity of previous example
# Now omit Z and plot
  # Make sure the range encompasses both data sets!
plot(x.z[,1], y, xlab="x",
     xlim=range(c(x.z[,1],new.x.z[,1])), ylim=range(c(y,new.y)))
rug(x.z[,1],side=1)
###axis(y,side=2)
points(new.x.z[,1],new.y,col="blue")
rug(new.x.z[,1],side=1,col="blue")
rug(new.y,side=2,col="blue")
# ... and regress
old.lm = lm(y ~ x.z[,1])
new.lm = lm(new.y ~ new.x.z[,1])
abline(old.lm)
abline(new.lm,col="blue")


## ----log-regression-curve-and-scatter, echo=FALSE-----------------------------
x <- runif(100)
y <- rnorm(100,mean=log(x),sd=1)
plot(y~x)
curve(log(x),add=TRUE,col="grey")
abline(lm(y~x))

## ----eval=FALSE---------------------------------------------------------------
## x <- runif(100)
## y <- rnorm(100,mean=log(x),sd=1)
## plot(y~x)
## curve(log(x),add=TRUE,col="grey")
## abline(lm(y~x))

