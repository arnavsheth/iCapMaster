# install packages
# install.packages("lavaan", dependencies = T)
# install.packages("remotes")
# remotes::install_github("bkeller2/fdir")
library(fdir)
library(lavaan)

# set working directory to this script's location
fdir::set()

# read data
dat <- read.table("mathcomplete.dat")
names(dat) <- c("id", "male", "lunchasst", "achievegrp", "stanread", "efficacy", "anxiety", "mathpre", "mathpost")

# data
Y <- dat$mathpost

# evaluate first derivatives at current parameter values
deriv.1 <- function(mu, var, Y){
  d.mu <- var^-1 * sum(Y - mu)
  d.var <- -N/2 * var^-1 + 1/2 * var^-2 * sum((Y - c(mu))^2)
  return(c(d.mu, d.var))}

# evaluate second derivatives at current parameter values
deriv.2 <- function(mu, var, Y){
  d.mu <- -N * var^-1
  d.var <- (N * .5 * var^-2) - var^-3 * sum((Y - c(mu))^2)
  return(c(d.mu, d.var))}

# initialize algorithmic features
N <- length(Y)
stopcriterion <- .000001
maxiterations <- 10000
mu.start <- 0
var.start <- 1
params <- cbind(rep(mu.start,maxiterations),rep(var.start,maxiterations))
logL <- rep(0,maxiterations)

# begin newton algorithm
iteration <- 1
maxdiff <- 1
logL[iteration] <- sum(dnorm(Y,params[iteration,1],sqrt(params[iteration,2]), log = T))
while (maxdiff > stopcriterion) {

  # advance iteration index
  iteration <- iteration + 1
  print(paste0("iteration = ", iteration))
  
  # evaluate first and second derivatives at the current parameter values
  l.deriv1 <- deriv.1(mu = params[iteration-1,1],var = params[iteration-1,2], Y = Y)
  l.deriv2 <- deriv.2(mu = params[iteration-1,1],var = params[iteration-1,2], Y = Y)
  
  # updating parameters
  params[iteration,1] <- params[iteration-1,1] - l.deriv1[1] / l.deriv2[1]
  params[iteration,2] <- params[iteration-1,2] - l.deriv1[2] / l.deriv2[2]
  
  # calculate max difference between consecutive parameters to assess convergence
  if(iteration < maxiterations){
    maxdiff <- max(abs(params[iteration,] - params[iteration-1,]))
  } else {maxdiff <- stopcriterion}
  
  # save log likelihood
  logL[iteration] <- sum(dnorm(Y,params[iteration,1],sqrt(params[iteration,2]), log = T))
  
}

# summarize estimates
results <- cbind(seq(1:(iteration))-1, logL[1:iteration], params[1:iteration,])
colnames(results) <- c("Iteration", "logL", "Mean", "Variance")
print(paste0("Estimation Summary after ", iteration-1, " Newton Iterations:"))
print(results, 15)

# check results with lavaan
library(lavaan)
model <- 'mathpost ~ 1; mathpost ~~ mathpost'
fit <- lavaan(model, dat)
lavaan::summary(fit, fit.measures = T)
