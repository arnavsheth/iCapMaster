# install packages
# install.packages("matrixcalc", dependencies = T)
# install.packages("lavaan", dependencies = T)
# install.packages("remotes")
# remotes::install_github("bkeller2/fdir")
options(scipen = 999)
library(fdir)
library(matrixcalc)

# set working directory to this script's location
fdir::set()

# read and select data
dat <- read.table("mplus.ex3.16.dat")
names(dat) <- c("m1","m2","y","x1","x2","x3")
Y <- cbind(dat$x1, dat$m1, dat$y)

# evaluate first derivatives at current parameter values
get.gradients <- function(mu, sigma){
  gradient.mu <- rep(0, num.vars)
  gradient.sigma <- rep(0, length(vech(sigma)))
  for(i in 1:nrow(Y)){
    kappa <- duplication.matrix(nrow(sigma))
    gradient.mu <- t(matrix((Y[i,] - mu), ncol = 1)) %*% solve(sigma) + gradient.mu
    gradient.sigma <- (.5*t(vec(solve(sigma) %*% matrix((Y[i,] - mu), ncol = 1) %*% t(matrix((Y[i,] - mu), ncol = 1)) %*% solve(sigma) - solve(sigma))) %*% kappa) + gradient.sigma
  }
  return(c(gradient.mu, gradient.sigma))}

# evaluate second derivatives at current parameter values
get.hessian <- function(mu, sigma){
  deriv.mu <- matrix(0, nrow = length(mu), ncol = length(mu))
  deriv.sigma <- matrix(0, nrow = ncol(duplication.matrix(nrow(sigma))), ncol = ncol(duplication.matrix(nrow(sigma))))
  deriv.mu.sigma <- matrix(0, nrow = num.vars, ncol = length(vech(sigma)))
  deriv.sigma.mu <- matrix(0, nrow = length(vech(sigma)), ncol = num.vars)
  for(i in 1:nrow(Y)){
    kappa <- duplication.matrix(nrow(sigma))
    deriv.mu <- solve(sigma) + deriv.mu
    deriv.sigma <- t(kappa) %*% (solve(sigma) %x% (solve(sigma) %*% matrix((Y[i,] - mu), ncol = 1) %*% t(matrix((Y[i,] - mu), ncol = 1)) %*% solve(sigma) - .5*solve(sigma))) %*% kappa + deriv.sigma
    deriv.mu.sigma <- (t(matrix((Y[i,] - mu), ncol = 1)) %*% solve(sigma) %x% solve(sigma)) %*% duplication.matrix(nrow(sigma)) + deriv.mu.sigma
    deriv.sigma.mu <- t(duplication.matrix(nrow(sigma))) %*% (solve(sigma) %*% matrix((Y[i,] - mu), ncol = 1) %x% solve(sigma)) + deriv.sigma.mu
      }
  hessian <- -rbind(cbind(deriv.mu, deriv.mu.sigma),cbind(deriv.sigma.mu,deriv.sigma))
  return(hessian)
}

# derivatives linking mediation model parameters to model-implied moments
# this is clunky for illustration, matrix expressions are in the literature
get.delta <- function(alpha, beta, psi){
  ix <- alpha[1]; im <- alpha[2]; iy <- alpha[3]
  apath <- beta[2,1]; gpath <- beta[3,1]; bpath <- beta[3,2]
  resx <- psi[1,1]; resm <- psi[2,2]; resy <- psi[3,3]
  delta <- matrix(0, nrow = length(mu) + length(vech(sigma)), ncol = num.parms)
  delta[1,1] <- 1;
  delta[2,1] <- apath; delta[2,2] <- 1; delta[2,4] <- ix
  delta[3,1] <- gpath + apath*bpath; delta[3,2] <- bpath; delta[3,3] <- 1; delta[3,4] <- bpath*ix; delta[3,5] <- ix; delta[3,6] <- im + apath*ix
  delta[4,7] <- 1
  delta[5,4] <- resx; delta[5,7] <- apath
  delta[6,4] <- bpath*resx; delta[6,5] <- resx; delta[6,6] <- apath*resx; delta[6,7] <- gpath + apath*bpath;
  delta[7,4] <- 2*apath*resx; delta[7,7] <- apath^2; delta[7,8] <- 1
  delta[8,4] <- (gpath + 2*apath*bpath)*resx; delta[8,5] <- apath*resx; delta[8,6] <- apath^2*resx + resm; delta[8,7] <- apath*gpath + apath^2*bpath; delta[8,8] <- bpath
  delta[9,4] <- (gpath + apath*bpath)*2*bpath*resx; delta[9,5] <- (gpath + apath*bpath)*2*resx; delta[9,6] <- 2*bpath*resm + 2*resx*(apath^2*bpath + apath*gpath); delta[9,7] <- gpath^2 + 2*apath*bpath*gpath + apath^2*bpath^2; delta[9,8] <- bpath^2; delta[9,9] <- 1
  return(delta)
}

# optimization settings
stopcriterion <- .000001
maxiterations <- 100
stepsize <- 1
diagreps <- 20 # initial reps use diagonal of hessian for updates

# mplus starting values
icept.start <- c(.046,-1.108,.499)
beta.start <- c(0,0, 0)
psi.start <- c(.572,8.716,15.879)

# initialize storage
num.vars <- length(icept.start)
num.parms <- length(icept.start) + length(beta.start) + length(psi.start)
params <- matrix(c(icept.start, beta.start, psi.start), nrow = 1) %x% matrix(1, nrow = maxiterations, ncol = 1)
logL <- rep(0,maxiterations)

# compute log-likelihood at starting values
alpha <- icept.start; beta <- matrix(0, num.vars, num.vars); beta[2,1] <- beta.start[1]; beta[3,1] <- beta.start[2]; beta[3,2] <- beta.start[3]; psi <- diag(psi.start)
mu <- solve(diag(num.vars) - beta) %*% alpha
sigma <- solve(diag(num.vars) - beta) %*% psi %*% t(solve(diag(num.vars) - beta))
logL[1] <- sum(dmvnorm(Y,mu,sigma, log = T))

# begin optimization
stepsize <- .15
iteration <- 1
maxdiff <- 1
while (maxdiff > stopcriterion) {

  # advance iteration index
  iteration <- iteration + 1
  print(paste0("starting iteration = ", iteration))
  
  # mediation model parameters from iteration t - 1
  alpha <- c(params[iteration-1,1:num.vars])
  beta[2,1] <- params[iteration-1,4]; beta[3,1] <- params[iteration-1,5]; beta[3,2] <- params[iteration-1,6]
  psi <- diag(params[iteration-1,(num.parms-2):num.parms])
  
  # model-implied mean vector and covariance matrix
  mu <- solve(diag(num.vars) - beta) %*% alpha
  sigma <- solve(diag(num.vars) - beta) %*% psi %*% t(solve(diag(num.vars) - beta))
  
  # delta matrix linking mu and sigma to mediation model parameters
  delta <- get.delta(alpha,beta,psi)
  
  if(iteration <= diagreps){ # update using diagonal of hessian and adaptively change stepsize
  
    # get gradient vector and hessian for model-implied moments at iteration t-1
    gradient <- get.gradients(mu,sigma)
    hessian <- get.hessian(mu,sigma)
    hessian <- diag(diag(hessian)) # use diagonal of hessian for initial reps
    
    # adaptively adjust step size to ensure that log-likelihood increases
    adjust.stepsize <- T
    while (adjust.stepsize == T) {
      
      params.new <- params[iteration-1,] - stepsize * solve(t(delta) %*% hessian %*% (delta)) %*% (t(delta) %*% gradient)
      
      # evaluate log-likelihood at updated estimates
      alpha <- c(params.new[1:num.vars])
      beta[2,1] <- params.new[4]; beta[3,1] <- params.new[5]; beta[3,2] <- params.new[6]
      psi <- diag(params.new[7:num.parms])
      mu.new <- solve(diag(num.vars) - beta) %*% alpha
      sigma.new <- solve(diag(num.vars) - beta) %*% psi %*% t(solve(diag(num.vars) - beta))
      logL.new <- sum(dmvnorm(Y,mu.new,sigma.new, log = T))
      logL.change <- logL.new - logL[iteration-1]
      
      # adaptively change step size to increase log-likelihood
      if(logL.change > 0){
        params[iteration,] <- params.new
        logL[iteration] <- logL.new
        stepsize <- stepsize * 1.25
        adjust.stepsize <- F
        print(paste0("iteration = ", iteration, "; logL(t-1) = ", logL[iteration-1], "; logL(t) = ", logL.new, "; logLcha = ", logL.change, "; stepsize = ", stepsize))
        } else {
        stepsize <- stepsize * .75
        print(paste0("iteration = ", iteration, "; logL(t-1) = ", logL[iteration-1], "; logL(t) = ", logL.new, "; logLcha = ", logL.change, "; stepsize = ", stepsize))
        }
      
    }
    
  } else { # update parameters using full hessian and increase step size
    
    stepsize <- 1
    
    # get gradient vector and hessian for model-implied moments at iteration t-1
    gradient <- get.gradients(mu,sigma)
    hessian <- get.hessian(mu,sigma)
    
    # updated estimates
    params.new <- params[iteration-1,] - stepsize * solve(t(delta) %*% hessian %*% (delta)) %*% (t(delta) %*% gradient)
    
    # evaluate log-likelihood at updated estimates
    alpha <- c(params.new[1:num.vars])
    beta[2,1] <- params.new[4]; beta[3,1] <- params.new[5]; beta[3,2] <- params.new[6]
    psi <- diag(params.new[7:num.parms])
    mu.new <- solve(diag(num.vars) - beta) %*% alpha
    sigma.new <- solve(diag(num.vars) - beta) %*% psi %*% t(solve(diag(num.vars) - beta))
    logL.new <- sum(dmvnorm(Y,mu.new,sigma.new, log = T))
    logL.change <- logL.new - logL[iteration-1]
    
    # save estimates
    params[iteration,] <- params.new
    logL[iteration] <- logL.new
    
    # summarize iteration history
    print(paste0("iteration = ", iteration, "; logL(t-1) = ", logL[iteration-1], "; logL(t) = ", logL.new, "; logLcha = ", logL.change, "; stepsize = ", stepsize))
    
  } # end else
  
  # calculate max difference between consecutive parameters to assess convergence
  if(iteration < maxiterations){
    maxdiff <- max(abs(params[iteration,] - params[iteration-1,]))
  } else {maxdiff <- stopcriterion}
  
  # compute standard errors after convergence
  if(maxdiff < stopcriterion){
    delta <- get.delta(alpha,beta,psi)
    hessian <- t(delta) %*% get.hessian(mu,sigma) %*% (delta)
    cov.params <- solve(-hessian)
    se.params <- sqrt(diag(cov.params))
  }
  
  
}

# summarize estimates
summary <- cbind(params[iteration,], se.params)
rownames(summary) <- c("icept.x","icept.m","icept.y","apath","gpath", "bpath","resvar.x", "resvar.m", "resvar.y")
colnames(summary) <- c("Est.", "Std. Err.")
iterhist <- cbind(seq(1:(iteration))-1, logL[1:iteration], params[1:iteration,])
colnames(iterhist) <- c("Iteration", "logL", "icept.x","icept.m","icept.y","apath","gpath", "bpath","resvar.x", "resvar.m", "resvar.y")

print(paste0("Estimation Summary after ", iteration-1, " Newton Iterations:"))
print(summary, 2)
print(paste0("Iteration History:"))
print(iterhist, 8)

# check results with lavaan
library(lavaan)
model <- 'x1 ~ 1; m1 ~ 1 + x1; y ~ 1 + x1 + m1;
          x1 ~~ x1; m1 ~~ m1; y ~~ y;'
fit <- lavaan(model, dat)
lavaan::summary(fit, fit.measures = T)


# my results
print(round(iterhist[nrow(iterhist),2], 3))

print(round(summary, 3))

