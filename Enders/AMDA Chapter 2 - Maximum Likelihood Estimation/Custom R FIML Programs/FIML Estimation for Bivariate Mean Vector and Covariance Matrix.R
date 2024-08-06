# install packages
# install.packages("matrixcalc", dependencies = T)
# install.packages("lavaan", dependencies = T)
# install.packages("remotes")
# remotes::install_github("bkeller2/fdir")

library(fdir)
library(matrixcalc)

# set the location of this script as the working directory
fdir::set()

# read and select data
dat <- read.table("employeecomplete.dat")
colnames(dat) <- c("employee","team","turnoverint","male","empower","lmx","jobsat","climate","cohesion")
Y <- cbind(dat$lmx, dat$empower)

# evaluate first derivatives at current parameter values
gradients <- function(mu, sigma){
  N <- nrow(Y)
  gradient.mu <- -N * solve(sigma) %*% mu + solve(sigma) %*% colSums(Y)
  gradient.sigma <- rep(0,ncol(duplication.matrix(nrow(sigma))))
  for(i in 1:N){
    gradient.sigma <- gradient.sigma + t(duplication.matrix(nrow(sigma))) %*% vec(-.5*(solve(sigma) - solve(sigma) %*% matrix((Y[i,] - mu), ncol = 1) %*% t(matrix((Y[i,] - mu), ncol = 1)) %*% solve(sigma)))
      }
  return(c(gradient.mu, gradient.sigma))}

# evaluate second derivatives at current parameter values
hessian <- function(mu, sigma){
  N <- nrow(Y)
  deriv.mu <- matrix(0, nrow = length(mu), ncol = length(mu))
  deriv.sigma <- matrix(0, nrow = ncol(duplication.matrix(nrow(sigma))), ncol = ncol(duplication.matrix(nrow(sigma))))
  deriv.mu.sigma <- matrix(0, nrow = length(mu), ncol = ncol(duplication.matrix(nrow(sigma))))
  for(i in 1:N){
    deriv.mu <- deriv.mu + solve(sigma)
    deriv.sigma <- deriv.sigma + t(duplication.matrix(nrow(sigma))) %*% (solve(sigma) %x% (solve(sigma) %*% matrix((Y[i,] - mu), ncol = 1) %*% t(matrix((Y[i,] - mu), ncol = 1)) %*% solve(sigma) - .5*solve(sigma))) %*% duplication.matrix(nrow(sigma))
    deriv.mu.sigma <- deriv.mu.sigma + (solve(sigma) %x% (t(matrix((Y[i,] - mu), ncol = 1)) %*% solve(sigma))) %*% duplication.matrix(nrow(sigma))
      }
  hessian <- -rbind(cbind(deriv.mu, deriv.mu.sigma),cbind(t(deriv.mu.sigma),deriv.sigma))
  return(hessian)}

# compute variance-covariance matrix of the ml estimates for illustration
cov.estimates <- solve(-hessian(mu,sigma))
# standard errors are sqrt of diagonal elements
se.estimates <- sqrt(diag(cov.estimates))

# initialize algorithmic features
stopcriterion <- .00001
maxiterations <- 100
mu.start <- rep(0,ncol(Y))
sigma.start <- diag(2)
num.vars <- length(mu.start)
num.parms <- length(mu.start) + length(vech(sigma.start))
params <- matrix(c(mu.start, vech(sigma.start)), nrow = 1) %x% matrix(1, nrow = maxiterations, ncol = 1)
logL <- rep(0,maxiterations)

# begin newton algorithm
iteration <- 1
maxdiff <- 1
while (maxdiff > stopcriterion) {

  # advance iteration index
  iteration <- iteration + 1
  print(paste0("iteration = ", iteration))
  
  # current parameters
  mu <- params[iteration-1,1:num.vars]
  sigma.vech <- params[iteration-1,(num.vars+1):num.parms]
  sigma <- matrix(duplication.matrix(length(mu)) %*% sigma.vech, ncol = num.vars)
  
  if(iteration <= 3){
    
    # update mu block (iteration <= 3)
    gradient.mu <- gradients(mu,sigma)[1:num.vars]
    hessian.mu <- hessian(mu,sigma)[1:num.vars,1:num.vars]
    params[iteration,1:num.vars] <- params[iteration-1,1:num.vars] - solve(hessian.mu) %*% gradient.mu
    
    # update sigma block (iteration <= 3)
    gradient.sigma <- gradients(mu,sigma)[(num.vars+1):num.parms]
    hessian.sigma <- hessian(mu,sigma)[(num.vars+1):num.parms,(num.vars+1):num.parms]
    
    params[iteration,(num.vars+1):num.parms] <- params[iteration-1,(num.vars+1):num.parms] - solve(hessian.sigma) %*% gradient.sigma
    
  } else {
    
    # update all parameters simultaneously (iteration > 3)
    params[iteration,] <- params[iteration-1,] - solve(hessian(mu,sigma)) %*% gradients(mu,sigma)
    
  }
  
  # calculate max difference between consecutive parameters to assess convergence
  if(iteration < maxiterations){
    maxdiff <- max(abs(params[iteration,] - params[iteration-1,]))
  } else {maxdiff <- stopcriterion}
  
  # save log likelihood
  logL[iteration-1] <- sum(dmvnorm(Y,mu,sigma, log = T))
  
  # compute standard errors after convergence
  if(maxdiff < stopcriterion){
    logL[iteration] <- sum(dmvnorm(Y,mu,sigma, log = T))
    cov.params <- solve(-hessian(mu,sigma))
    se.params <- sqrt(diag(cov.params))
  }
  
}

# summarize estimates
summary <- cbind(params[iteration,], se.params)
logL <- cbind(seq(1:(iteration))-1, logL[1:iteration], params[1:iteration,])
rownames(summary) <- c("mu.lmx","mu.empower","var.lmx","cov", "var.empower")
colnames(summary) <- c("Est.", "Std. Err.")
colnames(logL) <- c("Iteration", "logL", "mu.lmx","mu.empower","var.lmx","cov", "var.empower")
print(paste0("Estimation Summary after ", iteration-1, " Newton Iterations:"))
print(summary, 5)
print(paste0("Iteration History:"))
print(logL, 15)

# check results with lavaan
library(lavaan)
model <- 'lmx ~ 1; empower ~ 1; lmx ~~ empower; lmx ~~ lmx; empower ~~ empower'
fit <- lavaan(model, dat)
lavaan::summary(fit, fit.measures = T)

# my results
print(round(logL[nrow(logL),2], 3))
print(round(summary, 3))

