## ----echo=FALSE---------------------------------------------------------------
library(knitr)
opts_chunk$set(size="small",background="white", highlight=FALSE, cache=TRUE,
               autodep=TRUE)






## ----basic_missing_post_deletion, echo=FALSE, message=FALSE-------------------
the.df.post.deletion <- na.omit(the.df)
plot(y ~ x, data=the.df.post.deletion, xlab="x", ylab="y", type="p",
     xlim=c(min(x), max(x)), ylim=c(0,1))
rug(side=1, x=the.df.post.deletion$x)
rug(side=2, x=the.df.post.deletion$y)
require(mgcv)
a.spline <- gam(y ~ s(x), data=the.df.post.deletion)
lines(the.df.post.deletion$x, fitted(a.spline), col="grey")

## ----eval=FALSE---------------------------------------------------------------
## the.df.post.deletion <- na.omit(the.df)
## plot(y ~ x, data=the.df.post.deletion, xlab="x", ylab="y", type="p",
##      xlim=c(min(x), max(x)), ylim=c(0,1))
## rug(side=1, x=the.df.post.deletion$x)
## rug(side=2, x=the.df.post.deletion$y)
## require(mgcv)
## a.spline <- gam(y ~ s(x), data=the.df.post.deletion)
## lines(the.df.post.deletion$x, fitted(a.spline), col="grey")


## -----------------------------------------------------------------------------
n <- 1000
test.scores <- runif(n, min=200, max=1600)
gpas <- 4*(test.scores-200)/1400 + rnorm(n, sd=0.5)


## ----eval=FALSE---------------------------------------------------------------
## library(faraway) # for ilogit
## n <- 50
## # X is uniform (put it in order for easy plotting)
## x <- sort(runif(n, min=0, max=100))
## # Y increases with X, though non-linearly
## y <- ilogit(0.05*(x-50)+rnorm(n, sd=1))
## # Missing-ness depends on the value of Y, high values => more missing
## prob.y.missing <- ilogit(50*logit(y))
## missing.y <- (rbinom(n=n, size=1, prob=prob.y.missing) == 1) # To make it Boolean
## y.obs <- y[!missing.y]
## x.obs <- x[!missing.y]
## the.df <- data.frame(x=x, y=ifelse(missing.y, NA, y), missing.y=missing.y)
## plot(y~x, data=the.df, xlab="x", ylab="y", ylim=c(0,1))
## rug(side=1, x=the.df$x)
## rug(side=2, x=the.df$y)

