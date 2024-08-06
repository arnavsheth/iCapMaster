# load packages
library(fdir)
library(mvtnorm)

# set working directory to this script's location
fdir::set()

# read data
dat <- read.table("mathcomplete.dat")
names(dat) <- c("id", "male", "lunchasst", "achievegrp", "stanread", "efficacy", "anxiety", "mathpre", "mathpost")

X <- cbind(1, dat$mathpre, dat$efficacy, dat$lunchasst)
Y <- dat$mathpost

# initialize algorithmic features
iterations <- 11000
burnin <- 1000
betas <- matrix(0, ncol = 4, nrow = iterations)
sigma2e <- matrix(1, iterations)

# gibbs sampler
for(t in 2:iterations){
  
  # print iteration history
  if(t %% 500 == 0){print(paste0("Iteration = " , t))}

  # regression coefficients, conditional on residual variance and missing values
  betahat <- solve(crossprod(X,X)) %*% crossprod(X,Y)
  covbeta <- solve(crossprod(X,X)) * sigma2e[t-1]
  betas[t,] <- rmvnorm(1, mean = betahat, sigma = covbeta)
  
  # sample variance, conditional on coefficients and missing values
  df <- length(Y) / 2
  sumofsq <- sum((Y - X %*% betas[t,])^2) / 2
  sigma2e[t] <- 1 / rgamma(1, df, rate = sumofsq)
  
}

# summarize posterior distributions
summary <- matrix(0, nrow = 5, ncol = 5)
rownames(summary) <- c("B0", "B1", "B2", "B3", "res. var.")
colnames(summary) <- c("Mean", "StdDev", "2.5%", "50%", "97.5%")

# summarize posterior distributiona
params <- cbind(betas, sigma2e)
for(p in 1:5){
  summary[p,1] <- mean(params[(burnin+1):iterations,p])
  summary[p,2] <- sd(params[(burnin+1):iterations,p])
  summary[p,3:5] <- quantile(params[(burnin+1):iterations,p], c(.025, .50, .975))
  plot(density(params[(burnin+1):iterations,p]), main = rownames(summary)[p], xlab = "Parameter Value")
}

print(paste0("Posterior Distribution Summary from ", iterations - burnin, " Iterations:"))
print(round(summary, 3))


