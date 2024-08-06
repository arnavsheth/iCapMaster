# example 2.13: fiml logistic and probit regression
# requires fdir and aod packages

# set working directory
fdir::set()

# read data from working directory
dat <- read.table("employeecomplete.dat")
names(dat) <- c("employee","team","turnover","male","empower","lmx","worksat","climate","cohesion")

# logistic regression model with wald test
logitreg <- glm(turnover ~ lmx + empower + male, data = dat, family = "binomial")
summary(logitreg)
aod::wald.test(b = coef(logitreg), Sigma = vcov(logitreg), Terms = 2:4)

# probit regression model with wald test
probitreg <- glm(turnover ~ lmx + empower + male, data = dat, family = "binomial"(link = "probit"))
summary(probitreg)
aod::wald.test(b = coef(probitreg), Sigma = vcov(probitreg), Terms = 2:4)