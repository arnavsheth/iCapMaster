# install packages
# install.packages("matrixcalc", dependencies = T)
# install.packages("lavaan", dependencies = T)
# install.packages("remotes")
# remotes::install_github("bkeller2/fdir")

library(fdir)
library(matrixcalc)
options(scipen = 999)

# set working directory to this script's location
fdir::set()

# read and select data
dat <- read.table("mplus.ex3.16.dat")
names(dat) <- c("m1","m2","y","x1","x2","x3")
Y <- cbind(dat$x1, dat$m1, dat$y)

# evaluate first derivatives at current parameter values
gradients <- function(mu, sigma){
  gradient.mu <- rep(0, num.vars)
  gradient.sigma <- rep(0, num.parms-num.vars)
  for(i in 1:nrow(Y)){
      kappa <- duplication.matrix(nrow(sigma))
      gradient.mu <- t(matrix((Y[i,] - mu), ncol = 1)) %*% solve(sigma) + gradient.mu
      gradient.sigma <- (.5*t(vec(solve(sigma) %*% matrix((Y[i,] - mu), ncol = 1) %*% t(matrix((Y[i,] - mu), ncol = 1)) %*% solve(sigma) - solve(sigma))) %*% kappa) + gradient.sigma
      }
  return(c(gradient.mu, gradient.sigma))}

# evaluate second derivatives at current parameter values
hessian <- function(mu, sigma){
  deriv.mu <- matrix(0, nrow = length(mu), ncol = length(mu))
  deriv.sigma <- matrix(0, nrow = ncol(duplication.matrix(nrow(sigma))), ncol = ncol(duplication.matrix(nrow(sigma))))
  deriv.mu.sigma <- matrix(0, nrow = num.vars, ncol = num.parms-num.vars)
  deriv.sigma.mu <- matrix(0, nrow = num.parms-num.vars, ncol = num.vars)
  for(i in 1:nrow(Y)){
    tau <- diag(nrow(sigma))
    kappa <- (tau %x% tau) %*% duplication.matrix(nrow(sigma))
    deriv.mu <- t(tau) %*% solve(sigma) %*% tau + deriv.mu
    deriv.sigma <- t(kappa) %*% (solve(sigma) %x% (solve(sigma) %*% matrix((Y[i,] - mu), ncol = 1) %*% t(matrix((Y[i,] - mu), ncol = 1)) %*% solve(sigma) - .5*solve(sigma))) %*% kappa + deriv.sigma
    deriv.mu.sigma <- (t(matrix((Y[i,] - mu), ncol = 1)) %*% solve(sigma) %x% solve(sigma)) %*% duplication.matrix(nrow(sigma)) + deriv.mu.sigma
    deriv.sigma.mu <- t(duplication.matrix(nrow(sigma))) %*% (solve(sigma) %*% matrix((Y[i,] - mu), ncol = 1) %x% solve(sigma)) + deriv.sigma.mu
    }
  hessian <- -rbind(cbind(deriv.mu, deriv.mu.sigma),cbind(deriv.sigma.mu,deriv.sigma))
  return(hessian)
}

# evaluate second derivatives at current parameter values
hessian.cp <- function(mu, sigma){

  # mu <- mu.start; sigma <- sigma.start
  gradient.mu <- rep(0, num.vars)
  gradient.sigma <- rep(0, num.parms-num.vars)
  gradient.crossprod <- matrix(0, num.parms, num.parms)
  for(i in 1:nrow(Y)){
    tau <- diag(nrow(sigma))
    kappa <- (tau %x% tau) %*% duplication.matrix(nrow(sigma))
    gradient.mu <- t(matrix((Y[i,] - mu), ncol = 1)) %*% solve(sigma) %*% tau + gradient.mu
    gradient.sigma <- (.5*t(vec(solve(sigma) %*% matrix((Y[i,] - mu), ncol = 1) %*% t(matrix((Y[i,] - mu), ncol = 1)) %*% solve(sigma) - solve(sigma))) %*% kappa) + gradient.sigma
    gradient.crossprod <- c(gradient.mu, gradient.sigma) %*% t(c(gradient.mu, gradient.sigma)) + gradient.crossprod
  }
  hessian <- gradient.crossprod / nrow(Y)
  return(hessian)
}


# optimization settings
stopcriterion <- .00001
maxiterations <- 200
stepsize <- 1
blockreps <- 5

# initialize values
mu.start <- c(0,0,0)
sigma.start <- diag(c(.572,8.716,15.879))
num.vars <- length(mu.start)
num.parms <- length(mu.start) + length(vech(sigma.start))
params <- matrix(c(mu.start, vech(sigma.start)), nrow = 1) %x% matrix(1, nrow = maxiterations, ncol = 1)
logL <- rep(0,maxiterations); logL[1] <- sum(dmvnorm(Y,mu.start,sigma.start, log = T))

# begin optimization
iteration <- 1
maxdiff <- 1
while (maxdiff > stopcriterion) {

  # advance iteration index
  iteration <- iteration + 1
  print(paste0("iteration = ", iteration))
  
  # parameters from iteration t - 1
  mu <- params[iteration-1,1:num.vars]
  sigma.vech <- params[iteration-1,(num.vars+1):num.parms]
  sigma <- matrix(duplication.matrix(length(mu)) %*% sigma.vech, ncol = num.vars)
  
  if(iteration <= blockreps){ # update mu and sigma blocks separately (iteration <= blockreps)
  
    gradient.mu <- gradients(mu,sigma)[1:num.vars]
    gradient.sigma <- gradients(mu,sigma)[(num.vars+1):num.parms]
    hessian.mu <- hessian(mu,sigma)[1:num.vars,1:num.vars]
    hessian.sigma <- hessian(mu,sigma)[(num.vars+1):num.parms,(num.vars+1):num.parms]
    
    mu.new <- params[iteration-1,1:num.vars] - (stepsize * solve(-hessian.mu) %*% gradient.mu)
    sigma.vech.new <- params[iteration-1,(num.vars+1):num.parms] - stepsize * solve(diag(diag(hessian.sigma))) %*% gradient.sigma
    sigma.new <- matrix(duplication.matrix(length(mu.new)) %*% sigma.vech.new, ncol = num.vars)
    logL.new <- sum(dmvnorm(Y,mu.new,sigma.new, log = T))
    
    # adjust step size to ensure that log-likelihood increases
    adjust.stepsize <- 0
    while (adjust.stepsize == 0) {
      
      mu.new <- params[iteration-1,1:num.vars] - (stepsize * solve(hessian.mu) %*% gradient.mu)
      sigma.vech.new <- params[iteration-1,(num.vars+1):num.parms] - stepsize * solve(diag(diag(hessian.sigma))) %*% gradient.sigma
      sigma.new <- matrix(duplication.matrix(length(mu.new)) %*% sigma.vech.new, ncol = num.vars)
      logL.new <- sum(dmvnorm(Y,mu.new,sigma.new, log = T))
      logL.change <- logL.new - logL[iteration-1]
      
      # adaptively change step size to increase log-likelihood
      if(logL.change > 0){
        params[iteration,1:num.vars] <- mu.new
        params[iteration,(num.vars+1):num.parms] <- sigma.vech.new
        logL[iteration] <- logL.new
        stepsize <- stepsize * 2
        adjust.stepsize <- 1
        print(paste0("iteration = ", iteration, "; logL(t-1) = ", logL[iteration-1], "; logL(t) = ", logL[iteration], "; logLcha = ", logL.change, "; stepsize = ", stepsize))
        } else {
        stepsize <- stepsize * .50
        print(paste0("iteration = ", iteration, "; logL(t-1) = ", logL[iteration-1], "; logL(t) = ", logL[iteration], "; logLcha = ", logL.change, "; stepsize = ", stepsize))
        }
      
    }
    
  } else { # update all parameters simultaneously (iteration > blockreps)
    
    # adjust step size to ensure that log-likelihood increases
    adjust.stepsize <- 0
    while (adjust.stepsize == 0) {
      
      params.new <- params[iteration-1,] - stepsize * solve(hessian(mu,sigma)) %*% gradients(mu,sigma)
      mu.new <- params.new[1:num.vars]
      sigma.vech.new <- params.new[(num.vars+1):num.parms]
      sigma.new <- matrix(duplication.matrix(length(mu.new)) %*% sigma.vech.new, ncol = num.vars)
      logL.new <- sum(dmvnorm(Y,mu.new,sigma.new, log = T))
      logL.change <- logL.new - logL[iteration-1]
      
      # adaptively change step size
      if(logL.change > 0){
        params[iteration,1:num.vars] <- mu.new
        params[iteration,(num.vars+1):num.parms] <- sigma.vech.new
        logL[iteration] <- logL.new
        stepsize <- stepsize * 2
        adjust.stepsize <- 1
        print(paste0("iteration = ", iteration, "; logL(t-1) = ", logL[iteration-1], "; logL(t) = ", logL[iteration], "; logLcha = ", logL.change, "; stepsize = ", stepsize))
        } else {
          stepsize <- stepsize * .50
        }
      
    } # while loop
  
    
  } # end else
  
  # calculate max difference between consecutive parameters to assess convergence
  if(iteration < maxiterations){
    maxdiff <- max(abs(params[iteration,] - params[iteration-1,]))
  } else {maxdiff <- stopcriterion}
  
  # compute standard errors after convergence
  if(maxdiff < stopcriterion){
    hessian.sat <- hessian(mu,sigma)
    cov.params <- solve(-hessian.sat)
    se.params <- sqrt(diag(cov.params))
  }
  
  
}

# summarize estimates
summary <- cbind(params[iteration,], se.params)
rownames(summary) <- c("mu.x1","mu.m1","mu.y","var.x1","cov.m1x1", "cov.yx1","var.m1", "cov.ym1", "var.y")
colnames(summary) <- c("Est.", "Std. Err.")
iterhist <- cbind(seq(1:(iteration))-1, logL[1:iteration], params[1:iteration,])
colnames(iterhist) <- c("Iteration", "logL", "mu.x1","mu.m1","mu.y","var.x1","cov.m1x1", "cov.yx1","var.m1", "cov.ym1", "var.y")

print(paste0("Estimation Summary after ", iteration-1, " Newton Iterations:"))
print(summary, 5)
print(paste0("Iteration History:"))
print(iterhist, 10)
# 
# check results with lavaan
library(lavaan)
model <- 'x1 ~ 1; m1 ~ 1; y ~ 1;
          x1 ~~ m1; x1 ~~ y; m1 ~~ y;
          x1 ~~ x1; m1 ~~ m1; y ~~ y;'
fit <- lavaan(model, dat)
lavaan::summary(fit, fit.measures = T)


# my results
print(round(iterhist[nrow(iterhist),2], 3))

print(round(summary, 3))

