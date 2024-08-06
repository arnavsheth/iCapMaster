# load package
# library(fdir)
library(mvtnorm)

# read data
dat <- read.table("/Users/asheth/Library/CloudStorage/OneDrive-iCapitalNetwork/GMAM/Git/Enders/AMDA Chapter 4 - Bayesian Estimation/Custom R MCMC Programs/mathcomplete.dat")
names(dat) <- c("id", "male", "lunchasst", "achievegrp", "stanread", "efficacy", "anxiety", "mathpre", "mathpost")

Y <- dat$mathpost

# initialize algorithmic features
set.seed(90291)
iterations <- 11000
burnin <- 1000
mu <- matrix(0, iterations)
sigma2 <- matrix(1, iterations)

# gibbs sampler
for(t in 2:iterations){
  
  # print iteration history
  if(t %% 500 == 0){print(paste0("Iteration = " , t))}
  
  # sample mean, conditional on variance
  mean <- mean(Y)
  variance <- sigma2[t-1] / length(Y)
  mu[t] <- rnorm(1, mean = mean, sd = sqrt(variance))
  
  # sample variance, conditional on mean
  df <- length(Y) / 2
  sumofsq <- sum((Y - mu[t])^2) / 2
  sigma2[t] <- 1 / rgamma(1, df, rate = sumofsq)
}

# summarize posterior distributions
summary <- matrix(0, nrow = 2, ncol = 5)
rownames(summary) <- c("muY", "sigma2")
colnames(summary) <- c("Mean", "StdDev", "2.5%", "50%", "97.5%")

# summarize posterior distribution of the mean
params <- cbind(mu, sigma2)
for(p in 1:2){
  summary[p,1] <- mean(params[(burnin+1):iterations,p])
  summary[p,2] <- sd(params[(burnin+1):iterations,p])
  summary[p,3:5] <- quantile(params[(burnin+1):iterations,p], c(.025, .50, .975))
  plot(density(params[(burnin+1):iterations,p]), main = rownames(summary)[p], xlab = "Parameter Value")
}

print(paste0("Posterior Distribution Summary from ", iterations - burnin, " Iterations:"))
print(round(summary, 3))