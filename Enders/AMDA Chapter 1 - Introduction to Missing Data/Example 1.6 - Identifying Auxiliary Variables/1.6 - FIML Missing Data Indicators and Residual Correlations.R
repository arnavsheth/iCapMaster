# identify auxiliary variables with missing data indicators and residual correlations

# load packages
library(fdir)
library(lavaan)
library(semTools)

# set working directory and load data
set()
dat <- read.table("pain.dat", na.strings = "999")
names(dat) <- c("id", "txgrp", "male", "age", "edugroup", "workhrs", "exercise", "paingrps", 
                "pain", "anxiety", "stress", "control", "depress", "interfere", "disability",
                paste0("dep", seq(1:7)), paste0("int", seq(1:6)), paste0("dis", seq(1:6)))

#################################################################
# IDENTIFY CORRELATES OF MISSINGNESS USING COHEN'S D EFFECT SIZE
#################################################################

# missing data indicators
dat$mis.depress <- is.na(dat$depress)
dat$mis.interf <- is.na(dat$interfere)
dat$mis.pain <- is.na(dat$pain)

# auxiliary variables regressed on missing data indicators
model.depress <- '
     age ~ mis.depress
     exercise ~ mis.depress
     anxiety ~ mis.depress
     stress ~ mis.depress
     control ~ mis.depress
     disability ~ mis.depress
'

model.interfere <- '
     age ~ mis.interf
     exercise ~ mis.interf
     anxiety ~ mis.interf
     stress ~ mis.interf
     control ~ mis.interf
     disability ~ mis.interf
'

model.pain <- '
     age ~ mis.pain
     exercise ~ mis.pain
     anxiety ~ mis.pain
     stress ~ mis.pain
     control ~ mis.pain
     disability ~ mis.pain
'

# estimate model in lavaan with standardized auxiliary variables (outcomes)
fit.depress <- sem(model.depress, dat, fixed.x = T, missing = "fiml")
standardizedSolution(fit.depress, type = "std.nox")

# estimate model in lavaan with standardized auxiliary variables (outcomes)
fit.interfere <- sem(model.interfere, dat, fixed.x = T, missing = "fiml")
standardizedSolution(fit.interfere, type = "std.nox")

# estimate model in lavaan with standardized auxiliary variables (outcomes)
fit.pain <- sem(model.pain, dat, fixed.x = T, missing = "fiml")
standardizedSolution(fit.pain, type = "std.nox")

#################################################################
# IDENTIFY AUXILIARY VARIABLES BASED ON RESIDUAL CORRELATIONS
#################################################################

# incomplete variables regressed on all other analysis variables
model.depress <- 'depress ~ interfere + pain'
model.interf <- 'interfere ~ pain + depress'
model.pain <- 'pain ~ depress + interfere'

# potential auxiliary variables
auxvars <- c("age","exercise","anxiety","stress","control","disability")

# estimate model with lavaan and semTools
fit.depress <- sem.auxiliary(model.depress, dat, fixed.x = F, missing = "fiml", aux = auxvars)
summary(fit.depress, rsquare = T, standardize = T)

# estimate model with lavaan and semTools
fit.interf <- sem.auxiliary(model.interf, dat, fixed.x = F, missing = "fiml", aux = auxvars)
summary(fit.interf, rsquare = T, standardize = T)

# estimate model with lavaan and semTools
fit.pain <- sem.auxiliary(model.pain, dat, fixed.x = F, missing = "fiml", aux = auxvars)
summary(fit.pain, rsquare = T, standardize = T)