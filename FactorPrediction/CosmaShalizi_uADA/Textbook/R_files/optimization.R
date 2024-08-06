## ----include=FALSE------------------------------------------------------------
##### _Advanced Data Analysis from an Elementary Point of View_ #####
# R code for the chapter "Optimization"
# Please do not re-distribute or use without attribution
# http://www.stat.cmu.edu/~cshalizi/ADAfaEPoV/




## -----------------------------------------------------------------------------
my.newton = function(f,f.prime,f.prime2,beta0,tolerance=1e-3,max.iter=50) {
  beta = beta0
  old.f = f(beta)
  iterations = 0
  made.changes = TRUE
  while(made.changes & (iterations < max.iter)) {
   iterations <- iterations +1
   made.changes <- FALSE
   new.beta = beta - f.prime(beta)/f.prime2(beta)
   new.f = f(new.beta)
   relative.change = abs(new.f - old.f)/old.f -1
   made.changes = (relative.changes > tolerance)
   beta = new.beta
   old.f = new.f
  }
  if (made.changes) {
    warning("Newton's method terminated before convergence")
  }
  return(list(minimum=beta,value=f(beta),deriv=f.prime(beta),
              deriv2=f.prime2(beta),iterations=iterations,
              converged=!made.changes))
}


## ----regression-through-origin-scatterplot,echo=FALSE-------------------------
x <- runif(n=100,min=-1,max=1)
beta.true <- 4
y <- beta.true*x + rt(n=100,df=2)
plot(y~x)
abline(0,beta.true,col="grey")
abline(lm(y~x),lty=2)



## ----mse-curve,echo=FALSE,out.width="0.5\\textwidth"--------------------------
demo.mse <- function(b) { return(mean((y-b*x)^2)) }
curve(Vectorize(demo.mse)(x),from=0,to=10,xlab=expression(beta),ylab="MSE")
rug(x=beta.true,side=1,col="grey")



## ----lambda-vs-constraint,echo=FALSE,width="0.5\\textwidth"-------------------
lambda.from.c <- function(c) { mean(x*y)/sqrt(c) - mean(x^2) }
curve(lambda.from.c(x),from=0,to=20,xlab="c",ylab=expression(lambda))
abline(h=0, lty="dotted")



## ----penalized-beta-and-beta-squared, echo=FALSE------------------------------
par(mfrow=c(2,1))
beta.from.lambda <- function(l) { return(mean(x*y)/(l+mean(x^2))) }
curve(beta.from.lambda(x),from=0,to=6,
  xlab=expression(lambda),ylab=expression(tilde(beta)))
curve(beta.from.lambda(x)^2,from=0,to=6,
  xlab=expression(lambda),ylab=expression(tilde(beta)^2))
par(mfrow=c(1,1))

