# example 2.12: fiml means, variances, and correlations
# requires fdir and lavaan packages

# set working directory
fdir::set()

# read data from working directory
dat <- read.table("employeecomplete.dat")
names(dat) <- c("employee","team","turnover","male","empower","lmx","worksat","climate","cohesion")

# specify model
model <- '
  worksat ~~ empower
  worksat ~~ lmx
  empower ~~ lmx
'

# estimate model in lavaan
fit <- lavaan::sem(model, dat, meanstructure = T, fixed.x = F)
summary(fit, standardize = T)
