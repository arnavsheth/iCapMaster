# example 2.10: multiple regression
# requires fdir and lavaan packages

# set working directory
fdir::set()

# read data from working directory
dat <- read.table("smokingcomplete.dat")
names(dat) <- c("id","intensity","hvysmoker","age","parsmoke","female","race","income","pareduc")

# center variables at their grand means
dat$age.cgm <- dat$age - mean(dat$age)
dat$income.cgm <- dat$income - mean(dat$income)

# specify model
model <- 'intensity ~ parsmoke + age.cgm + income.cgm'

# regression model
fit <- lavaan::sem(model, dat, meanstructure = T, fixed.x = T)
lavaan::summary(fit, rsquare = T, standardize = T)

# bootstrap standard errors and test statistics
fit <- lavaan::sem(model, dat, meanstructure = T, fixed.x = T, se = "bootstrap")
lavaan::summary(fit, rsquare = T, standardize = T)

# robust standard errors and test statistics
fit <- lavaan::sem(model, dat, meanstructure = T, fixed.x = T, estimator = "MLR")
lavaan::summary(fit, rsquare = T, standardize = T)

# specify model with labels
model.labels <- 'intensity ~ b1*parsmoke + b2*age.cgm + b3*income.cgm'
wald.constraints <- 'b1 == 0; b2 == 0; b3 == 0'

# wald test that all slopes equal 0
fit <- lavaan::sem(model.labels, dat, meanstructure = T, fixed.x = T, estimator = "MLR")
lavaan::summary(fit, rsquare = T, standardize = T)
lavaan::lavTestWald(fit, constraints = wald.constraints)

# specify model with constraints
model.constraints <- 'intensity ~ b1*parsmoke + b2*age.cgm + b3*income.cgm
                      b1 == 0; b2 == 0; b3 == 0'

# constraining slopes to 0 gives LRT versus the saturated model
fit <- lavaan::sem(model.constraints, dat, meanstructure = T, fixed.x = T, estimator = "MLR")
lavaan::summary(fit, rsquare = T, standardize = T)


