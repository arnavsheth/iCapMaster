#################################################################################################
#  This is an example based on the data used in Alicia H. Munnell, Geoffrey M. B. Tootell, 
# Lynn E. Browne and James McEneaney (1996, AER) to measure the effect of being back on the
# probability of mortgage denial.  
#  14.382 L6 MIT.  V. Chernozhukov, I. Fernandez-Val
#  Credit to: A. Timoshenko (who fixed an error in the code).
#
# Data source: Boston HMDA Data Set (Stock and Watson version)
# URL for data: http://wps.aw.com/aw_stock_ie_3/178/45691/11696965.cw/index.html/

# Description of the data: the sample selection and variable contruction follow

# Alicia H. Munnell, Geoffrey M. B. Tootell, Lynn E. Browne and James McEneaney (1996) 
# "Mortgage Lending in Boston: Interpreting HMDA Data,"  The American Economic Review Vol. 86, No. 1 (Mar., 1996), pp. 25-53 

# Sample selection: single-residences (excluding data on multifamily homes), 2,380 observation

# The variables in the data set include:

# deny:  = 1 if mortgage application denied
# p_irat: montly debt to income ratio
# black: = 1 if applicant is black
# hse_inc: montly housing expenses to income ratio
# loan_val: loan to assessed property value ratio (not used in analysis)
# ccred: consumer credit score 
#        = 1 if no "slow" payments or delinquencies
#        = 2 if one or two "slow" payments or delinquencies
#        = 3 if more than two "slow" payments or delinquencies
#        = 4 if insufficient credit history for determination
#        = 5 if delinquent credit history with payments 60 days overdue
#        = 6 if delinquent credit history with payments 90 days overdue
# mcred: mortgage credit score
#        = 1 if no late mortgage payments
#        = 2 if no mortgage payment history
#        = 3 if one or two late mortgage payments
#        = 4 if more than two late mortgage payments
# pubrec: = 1 if any public record of credit problems (bankruptcy, charge-offs, collection actions)
# denpmi: = 1 if applicant applied for mortgage insurance and was denied
# selfemp: = 1 if self-employed
# single: = 1 if single
# hischl: = 1 if high school graduated
# probunmp: 1989 Massachusetts unemployment rate in the applicant's industry (not used in analysis)
# condo: = 1 if unit is a condominium (not used in analysis)
# ltv_med: = 1 if medium loan to property value ratio [.80, .95]
# ltv_high: = 1 if high loan to property value ratio > .95
#################################################################################################


# Updated on 03/26/2015


##################################################################################################



### set working directory
setwd("/Users/VC/Dropbox/Teaching/14.382/")
setwd("/Users/Ivan/Dropbox/Shared/14.382/")
###  read-in Mortgage data

library(foreign);
library(xtable);

data.In<- read.dta("Data/Mortgages/mortgage.dta")
n1<- dim(data.In)[1]

########## Descriptive statistics

options(digits=2);
dstats <- cbind(sapply(data.In, mean), apply(data.In[data.In$black==1,], 2, mean), apply(data.In[data.In$black==0,], 2, mean));
xtable(dstats);

########## Basic ols regressions

fmla1 =  deny ~ black + p_irat + hse_inc + ccred + mcred  + pubrec + ltv_med + ltv_high + denpmi + selfemp + single + hischl                  
fit.1=lm(fmla1, data=data.In)

fmla2 =  deny ~ black 
fit.2=lm(fmla2, data=data.In)

xtable(summary(fit.2, digits=3))
xtable(summary(fit.1, digits=3))


########## here we will split the data in two parts, data 1 and data 2

set.seed(3)
validate<- sample(1:n1, floor(n1/3), replace=F)

data1<- data.In[-c(validate),]  

### data 1 is the main part

data2<- data.In[c(validate),]

### data 2 is the validation part, which will be used to select the link function
### you can also use to choose better predictive models


### the following command attaches data1, making its columns directly accessible by their names

attach(data1)

################ Part 1: Compare some basic models #########################################################
#basic linear probability model, linear index

fmla1 =  deny ~ black + p_irat + hse_inc + ccred + mcred  + pubrec + ltv_med + ltv_high + denpmi + selfemp + single + hischl                  
fit.1=lm(fmla1)

#### basic models with logistic, probit, and cauchit links

fit.lgt.1=glm(fmla1, family=binomial(link="logit"))
fit.prt.1=glm(fmla1, family=binomial(link="probit"))
fit.cat.1=glm(fmla1, family=binomial(link="cauchit"))

pdf("Results/LogitvsLinear1_mort.pdf", pointsize=15,width=6,height=6)
plot(predict(fit.lgt.1, type="response"),predict(fit.1, type="response"),
     xlim=c(0,1), ylim=c(-0.1,1.1), xlab="logit prediction", ylab="linear & cauchit prediction", 
     main="Comparison of Predicted Probabilities", type="p", pch=3,col=2)
lines(predict(fit.lgt.1, type="response"),predict(fit.cat.1, type="response"),
     type="p", pch=2, col=3)
abline(0, 1)
abline(h=1)
abline(h=0)
dev.off()



pdf("Results/LogitvsProbit1_mort.pdf", pointsize=15,width=6,height=6)
plot(predict(fit.lgt.1, type="response"),predict(fit.prt.1, type="response"),
     xlim=c(0,1), ylim=c(0,1), xlab="logit prediction", ylab="probit prediction", 
     main="Comparison of Predicted Probabilities",col=4)
abline(0, 1)
dev.off()



# compare out-of-sample prediction scores

table.P<- matrix(0, ncol=4, nrow=1)
table.P[1,1]<- sqrt(mean((data2[,"deny"]- predict(fit.lgt.1, data2, type="response"))^2))
table.P[1,2]<- sqrt(mean((data2[,"deny"]- predict(fit.prt.1, data2, type="response"))^2))
table.P[1,3]<- sqrt(mean((data2[,"deny"]- predict(fit.cat.1, data2, type="response"))^2))
table.P[1,4]<- sqrt(mean((data2[,"deny"]- predict(fit.1, data2, type="response"))^2))

colnames(table.P)<- c("Logit", "Probit", "Cauchit", "Linear")
rownames(table.P)<- c("Mean Square Prediction Error")


xtable(table.P, digits=4, align=c(rep("c", 5)))


# logistic seems to work weakly better than others in terms of predicting

################################# Part 2. Predicted Effects of Black##################

###############################################################################################
# logistic model
###############################################################################################

## Reestimate model with entire sample
detach(data1)
attach(data.In)
n <- n1;
fit.lgt.1=glm(fmla1, family=binomial(link="logit"))
xtable(fit.lgt.1)   #report a nice table

## black=0 desing matrix
d0<- fit.lgt.1$model;  d0[,2]<- rep(0,n)
## black = 1 design matrix
d1<- fit.lgt.1$model;  d1[,2]<- rep(1,n)

impact<- predict(fit.lgt.1, newdata=d1, type="response")-  
  predict(fit.lgt.1, newdata=d0, type="response");
mean(impact); sqrt(var(impact));

pdf("Results/PredictedEffectLogitProbModel_mort.pdf", pointsize=15,width=8.0,height=8.0)
plot(0:100, quantile(impact, 0:100/100), type="l",col=4, xlab="percentile", 
     ylab="Effect of Black", 
     main= "Effects on Probabilities of Mortgage Denial",
     sub="Logit Model")
abline(h=mean(impact),col=3)
legend(20,.15, c(paste("mean effect ="), round(mean(impact),digits=3) ))
mean(impact)
dev.off()


#################################  Predicted Effects ############################
# cauchy model
#################################################################################

## Reestimate model with entire sample
fit.cat.1=glm(fmla1, family=binomial(link="cauchit"))
d0<- fit.cat.1$model;  d0[,2]<- rep(0,n)
d1<- fit.cat.1$model;  d1[,2]<- rep(1,n)
impact<- predict(fit.cat.1, newdata=d1, type="response")-  predict(fit.cat.1, newdata=d0, type="response");
mean(impact); sqrt(var(impact));
pdf("Results/PredictedEffectCauchyProbModel_mort.pdf", pointsize=15,width=8.0,height=8.0)
plot(1:100, quantile(impact, 1:100/100), type="l",col=4, xlab="percentile", 
     ylab="Effect of Black", 
     main= "Effects on Probabilities of Mortgage Denial",
     sub= "Cauchit Model")
abline(h=mean(impact),col=3)
legend(20,.15, c(paste("mean effect ="), round(mean(impact),digits=3) ))
mean(impact)
dev.off()

#################################  Predicted Effects ###########################################
# linear model
###############################################################################################

## Reestimate model with entire sample
fit.1=lm(fmla1)

## black=0 desing matrix
d0<- fit.1$model;  d0[,2]<- rep(0,n)

## black = 1 design matrix
d1<- fit.1$model;  d1[,2]<- rep(1,n)

impact<- predict(fit.1, newdata=d1, type="response")-  predict(fit.1, newdata=d0, type="response");
mean(impact); sqrt(var(impact));
pdf("Results/PredictedEffectLinearProbModel_mort.pdf", pointsize=15,width=8.0,height=8.0)
plot(1:100, quantile(impact, 1:100/100), type="l",col=4, xlab="percentile", 
     ylab="Effect of Black", 
     main= "Effects on Probabilities of Mortgage Denial",
     sub="Linear Model", 
      ylim=c(0,.25)
)

abline(h=mean(impact),col=3)
legend(20,.15, c(paste("mean effect ="), round(mean(impact),digits=3) ))
mean(impact)
dev.off()


################################ Part 3. BOOTSTRAP THE APE and SPE Logit Case ###########################

library(boot)  #imports the boot library

alpha  <- .10 #Significance level
########## statistic to be computed in each bootstrap draw #####################################

boot.PE<- function(data,indices){
data.b<- data[indices,];  
fmla1 =  deny ~ black + p_irat + hse_inc + ccred + mcred  + pubrec + ltv_med + ltv_high + denpmi + selfemp + single + hischl                  
#basic logistic model, linear index
fit.lgt.1=glm(fmla1, data=data.b, family=binomial(link="logit"))
## black=0 design matrix
n<-dim(data.b)[1]
d0<- fit.lgt.1$model;  d0[,2]<- rep(0,n)
## black = 1 design matrix
d1<- fit.lgt.1$model;  d1[,2]<- rep(1,n)
impact<- quantile( predict(fit.lgt.1, newdata=d1, type="response")-  
  predict(fit.lgt.1, newdata=d0, type="response"), 0:100/100);
return(impact)
}

################### application of bootstrap ##################################################

set.seed(1)

# bootstrap using parallelization

result.boot.PE<- boot(data=data.In, statistic=boot.PE, parallel="multicore", ncpus = 24, R=200)



################## estimate SPE and processes BS results #######################################
#
#
# black=0 design matrix
d0<- fit.lgt.1$model;  d0[,2]<- rep(0,n)
# black=1 design matrix
d1<- fit.lgt.1$model;  d1[,2]<- rep(1,n)

est.SPE<- quantile( 
            predict(fit.lgt.1, newdata=d1, type="response")-  
            predict(fit.lgt.1, newdata=d0, type="response"),
            0:100/100);   # these are estimated SPEs

draws.SPE<- result.boot.PE$t  #these are bs draws

# centered/scaled draws
delta    <- draws.SPE-matrix(est.SPE, nrow = nrow(draws.SPE), ncol = ncol(draws.SPE), byrow=TRUE)
variance <- apply(delta*delta,2,mean)
zs<- apply(abs(delta /matrix(sqrt(variance), nrow = nrow(draws.SPE), ncol = ncol(draws.SPE), byrow=TRUE)), 1, max)  #max abs t-stat
crt<- quantile(zs, 1-alpha)  #critical value

ubound.SPE<-  est.SPE + crt*sqrt(variance)  #upper conf band
lbound.SPE<-  est.SPE - crt*sqrt(variance)  #lower conf band


################# Part 3. Estimate the Averate Partial Effects and process BS results ########################

est.APE<- mean(est.SPE)  #compute as average of SPE
draws.APE<- apply(draws.SPE, 1, mean)    #bs draws of APEs
mzs<- abs(draws.APE-est.APE)/sqrt(var(draws.APE))   # abs t-stat
crt2<- quantile(mzs, 1-alpha)   #critical value

ubound.APE<-  est.APE+ crt2*sqrt(var(draws.APE))  #upper conf bound
lbound.APE<- est.APE - crt2*sqrt(var(draws.APE))  #lower conf bound



pdf("Results/PredictedEffectInference_mort.pdf", pointsize=15,width=8.0,height=8.0)
plot(0:100, est.SPE, type="l",col=4, xlab="percentile", lwd=2,
     ylab="Effect of Black", 
     main="Predictive Effects on Probabilities of Mortgage Denial",
     sub="Logistic Model", 
     ylim=c(min(lbound.SPE), max(ubound.SPE)))

polygon(c(0:100,rev(0:100)),c(ubound.SPE,rev(lbound.SPE)), density=60, border=F, 
        col='light blue', lty = 1, lwd = 1);
lines(0:100, est.SPE, lwd=2  )

lines(0:100, est.SPE, lwd=2, col=1 )
lines(0:100, rep(est.APE,101), lwd=2,col=2)
polygon(c(0:100,rev(0:100)),c(rep(ubound.APE,101),rep(lbound.APE,101)), 
        density=0, border=T, col=2, lty =1 , lwd = 2);

dev.off();



#################### Part 4: Whose probabilities are most affected? ################################

effect<-  predict(fit.lgt.1, newdata=d1, type="response")- predict(fit.lgt.1, newdata=d0, type="response")
high.effect<- quantile(effect, .95)
low.effect<- quantile(effect, .05)
print(high.effect); print(low.effect);

colnames<- names(d1)
high.affected<- data.In[effect>=high.effect,colnames]
low.affected<- data.In[effect<=low.effect,colnames]

# look at two characterisitic only

table.who<- matrix(0,2,length(colnames))
colnames(table.who)<- colnames
table.who[1,]<- apply(high.affected, 2, mean)
table.who[2,]<- apply(low.affected, 2, mean)

xtable(t(table.who), digits=2)



mean(deny[black==1])
#[1] 0.2831858
mean(deny[black==0])
#[1] 0.09260167



######################################################################################################



################################# Part 5. Countefactual Effects of Being Non-Black for Black ##################

###############################################################################################
# logistic model
###############################################################################################

## Reestimate model with entire sample
data.bl<- subset(data.In, black==1)
n <- dim(data.bl)[1]
fit.lgt.1=glm(fmla1, family=binomial(link="logit"))
xtable(fit.lgt.1)   #report a nice table

## desing matrix
d0<- fit.lgt.1$model[which(black==1),];  #black
d1<- d0; d1[,2]<- rep(0,n);  # black --> non-black


################################ Part 3. BOOTSTRAP THE APE and SPE Logit Case ###########################

library(boot)  #imports the boot library

alpha  <- .10 #Significance level
########## statistic to be computed in each bootstrap draw #####################################

boot.PE2<- function(data,indices){
  data.b<- data[indices,];  
  fmla1 =  deny ~ black + p_irat + hse_inc + ccred + mcred  + pubrec + ltv_med + ltv_high + denpmi + selfemp + single + hischl                    
  #basic logistic model, linear index
  fit.lgt.1=glm(fmla1, data=data.b, family=binomial(link="logit"))
  n <- sum(data.b[,c("black")])
  black.b<- data.b[,c("black")]
  ## desing matrix
  d0<- fit.lgt.1$model[which(black.b==1),];  #black
  d1<- d0; d1[,2]<- rep(0,n);  # black --> non-black
  impact<- quantile( -predict(fit.lgt.1, newdata=d1, type="response")+ 
                       predict(fit.lgt.1, newdata=d0, type="response"), 0:100/100);
  return(impact)
}

################### application of bootstrap ##################################################

set.seed(1)

# bootstrap using parallelization

result.boot.PE2<- boot(data=data.In, statistic=boot.PE2, parallel="multicore", ncpus = 24, R=200)



################## estimate SPE and processes BS results #######################################



est.SPE2<- quantile( 
  -predict(fit.lgt.1, newdata=d1, type="response")+  
    predict(fit.lgt.1, newdata=d0, type="response"),
  0:100/100);   # these are estimated SPEs

draws.SPE2<- result.boot.PE2$t  #these are bs draws

# centered/scaled draws

delta    <- draws.SPE2-matrix(est.SPE2, nrow = nrow(draws.SPE2), ncol = ncol(draws.SPE2), byrow=TRUE)
variance <- apply(delta*delta,2,mean)
zs<- apply(abs(delta /matrix(sqrt(variance), nrow = nrow(draws.SPE2), ncol = ncol(draws.SPE2), byrow=TRUE)), 1, max)  #max abs t-stat
crt<- quantile(zs, 1-alpha)  #critical value

ubound.SPE2<-  est.SPE2 + crt*sqrt(variance)  #upper conf band
lbound.SPE2<-  est.SPE2 - crt*sqrt(variance)  #lower conf band


# zs<- apply(abs((draws.SPE2-matrix(est.SPE2, nrow = nrow(draws.SPE2), ncol = ncol(draws.SPE2), byrow=TRUE))
#                /matrix(sqrt(apply(draws.SPE2, 2, var)), nrow = nrow(draws.SPE2), ncol = ncol(draws.SPE2), byrow=TRUE)), 1, max)  #max abs t-stat
# crt<- quantile(zs, 1-alpha)  #critical value
# 
# ubound.SPE2<-  est.SPE2 + crt*sqrt(apply(draws.SPE2, 2, var))  #upper conf band
# lbound.SPE2<-  est.SPE2 - crt*sqrt(apply(draws.SPE2, 2, var))  #lower conf band


#################Estimate the Averate Partial Effects and process BS results ########################

est.APE2<- mean(est.SPE2)  #compute as average of SPE
draws.APE2<- apply(draws.SPE2, 1, mean)    #bs draws of APEs
mzs<- abs(draws.APE2-est.APE2)/sqrt(var(draws.APE2))   # abs t-stat
crt2<- quantile(mzs, 1-alpha)   #critical value

ubound.APE2<-  est.APE2+ crt2*sqrt(var(draws.APE2))  #upper conf bound
lbound.APE2<- est.APE2 - crt2*sqrt(var(draws.APE2))  #lower conf bound

pdf("Results/PredictedEffect2Inference_mort.pdf", pointsize=14,width=8.0,height=8.0)
plot(0:100, est.SPE2, type="l",col=4, xlab="percentile", lwd=2,
     ylab="Effect of Black", 
     main="Predictive Effects on Probabilities of Mortgage Denial, for Blacks",
     sub="Logistic Model", 
     ylim=c(min(lbound.SPE2), max(ubound.SPE2)))

polygon(c(0:100,rev(0:100)),c(ubound.SPE2,rev(lbound.SPE2)), density=60, border=F, 
        col='light blue', lty = 1, lwd = 1);
lines(0:100, est.SPE2, lwd=2  )

lines(0:100, est.SPE2, lwd=2, col=1 )
lines(0:100, rep(est.APE2,101), lwd=2,col=2)
polygon(c(0:100,rev(0:100)),c(rep(ubound.APE2,101),rep(lbound.APE2,101)), 
        density=0, border=T, col=2, lty =1 , lwd = 2);

dev.off();




