#################################################################################################
#  14.382 L1 MIT.  V. Chernozhukov, I. Fernandez-Val

#' Illustration of Frisch-Waugh-Lovell Theorem using CPS earnings data for 2015
#' We are interested in wage discrimination for women here.
 
# Data source: IPUMS-CPS through Mert Demirer
# URL for data: https://cps.ipums.org/cps/

# See attached codebook for the transformed CPS variables and description of the constructed variables.

# The variables used in the analysis include:

# - lnw: log of hourly wage (annual earnings / annual hours)
# - female: female indicator
# - 6 married status indicators: widowed, divorced, separated, nevermarried, and married 
# - 5 education attainment indicators: lhs, hsg, cg, ad, and sc 
# - 4 region indicators: mw, so, we, and ne 
# - Quartic in potential experience (max[0, age - years of education - 7]): exp1, exp2 (divided by 100), exp3 (divided by 1000), exp4 (divided by 10000)
# - weight: March Supplement sampling weight
# - year: CPS year
# - Occupation categorical variable with 22 categories
# - Industry categorical variable with 22 categories

# Updated on 02/07/2017

# Loading data and packages

library(sandwich); # Robust standard errors
library(xtable)    # Latex tables

rm(list=ls())	 #clean memory

filepathIvan<-"/Users/Ivan/Dropbox/Shared/14.382"
filepathVictor<-"/Users/vchern/Dropbox/TEACHING/14.382"
#setwd("/Users/VC/Dropbox/TEACHING/14.382/L1/");
#setwd(filepathIvan)
setwd(filepathVictor)

#load('Data/CPS/cps_all.rdata');	
#data <- cps;	
options(warn=-1); # sets warnings off;
load('Data/CPS2015/cps2015.rdata');
save(data, file="Data/CPS2015/cps2015.rdata")
write.csv(data, file="Data/CPS2015/cps2015.csv")

# create education factor;
data$educ <- factor(data$lhs+2*data$hsg+3*data$sc+4*data$cg+5*data$ad);

# create marital status factor;
data$ms <- factor(data$married+2*data$widowed+3*data$divorced+4*data$nevermarried);

# create region;
data$reg <- factor(data$ne+2*data$mw+3*data$so+4*data$we);


attach(data);

# We start showing some descriptive statistics;

vars <- c("lnw","female","married","widowed","separated","divorced","nevermarried","lhs","hsg","sc","cg","ad","ne","mw","so","we",
          "exp1");

dstats <- cbind(sapply(data[,vars], mean), apply(data[female==0,vars], 2, mean), apply(data[female==1,vars], 2, mean));
colnames(dstats) <- c("All", "Male", "Female");
xtable(dstats, digits = 2);


#' - We first estimate the usual least squares:	
# fmla<- lnw ~ factor(year) + 
#   female + widowed + divorced + separated + nevermarried  + 
#   hsd08+hsd911+ hsg+cg+ad+mw+so+we+exp1+exp2+exp3+exp4	

#fmla<- lnw ~ female + widowed + divorced + separated + nevermarried +(hsd08+hsd911+hsg+cg+ad)*(exp1+exp2+exp3+exp4) + mw + so + we;
#fmla <- lnw ~ female + widowed + divorced + separated + nevermarried + (exp1+exp2+exp3+exp4)*(educ+occ2+ind2) + (educ+occ2+ind2)^2 + mw + so + we;
fmla <- lnw ~ female + (exp1+exp2+exp3+exp4)*(educ+occ2+ind2+ms+reg) + (educ+occ2+ind2+ms+reg)^2;

full.fit<- lm(fmla);
HCV.coefs <- vcovHC(full.fit, type = 'HC');
full.se <- sqrt(diag(HCV.coefs)); # White std errors


#'  - This runs a regression of log-wage ($Y$) on female indicator ($D$) and controls	
#' ($X$), which represent education levels, year dummies, quartic term in experience, and some	
#' other job-relevant characteristics.	

 	
#'  Then we do the partialling-out:	
#' 	
#' 	
#fmla.y<- lnw~ widowed + divorced + separated + nevermarried + (exp1+exp2+exp3+exp4)*(educ+occ2+ind2) + (educ+occ2+ind2)^2 + mw + so + we;	
#fmla.d<- female~widowed + divorced + separated + nevermarried + (exp1+exp2+exp3+exp4)*(educ+occ2+ind2) + (educ+occ2+ind2)^2 + mw + so + we;	
fmla.y<- lnw~ (exp1+exp2+exp3+exp4)*(educ+occ2+ind2+ms+reg) + (educ+occ2+ind2+ms+reg)^2;	
fmla.d<- female~(exp1+exp2+exp3+exp4)*(educ+occ2+ind2+ms+reg) + (educ+occ2+ind2+ms+reg)^2;	
rY<- lm(fmla.y)$res;	
rD<- lm(fmla.d)$res;	
partial.fit<- lm(rY~rD);
HCV.coefs <- vcovHC(partial.fit, type = 'HC');
partial.se <- sqrt(diag(HCV.coefs)); # White std errors



#' - Here "rY" is the name of the parialled out $Y$, i.e. the residual that is left from partialling out $X$	
#' - Here "rD" is the name of the partialled out $D$, i.e. the residual that is left from 	
#' partialling out $X$.	

#' - We then regress "rY" on "rD".	

#' - Then we compare the results from full regression and partial regression in a table.	
#' 	
#' 	
table<- matrix(0, 2, 2)	
table[1,1]<- summary(full.fit)$coef["female",1]   #extract coeff on female indicator	
table[1,2]<- full.se["female"]   #extract robust se on female indicator  
table[2,1]<- summary(partial.fit)$coef["rD",1]   #extract coeff on female indicator  
table[2,2]<- partial.se["rD"]   #extract robust se on female indicator  
colnames(table)<- names(summary(full.fit)$coef["female",])[1:2]	
rownames(table)<- c("full reg", "partial reg")	
tab<- xtable(table, digits=c(2, 5,10))	
#' 	
#' ---	
#' 	
#' # Compare the full to partial regression	
#' 	
#' 	
print(tab, type="latex")	

#' - This prints a nice latex table that you can insert in the document
#' 
#' - The point estimates are numerically equivalent, which is in line with the Frisch-Waugh-Lovell theorem.	
#' 	
#' - The standard errors are extremely  close (though not equivalent), which is in line with the asymptotics where the number of controls is much smaller than the sample size,  $p << n$.	
#' 	
#' 	
#' ---	
#' 	
#' # What if we try (post) LASSO method for partialling out? 	
#' 	
#' - We load a required "hdm" package:	
# install.packages("hdm")	
#library("hdm")	

#' ---	
#' 	
#' - Then we do partialling out using (post) lasso.  The "hdm" package uses 
#' the post-lasso by default.	
#' 	
#' 	
#' 	
library(hdm)	
rY<- rlasso(fmla.y)$res	
rD<- rlasso(fmla.d)$res	
partial.fit.lasso<- lm(rY~rD)	
HCV.coefs <- vcovHC(partial.fit.lasso, type = 'HC');
partial.lasso.se <- sqrt(diag(HCV.coefs)); # White std errors

#' - This is a (version of) the method proposed in Belloni, Chernozhukov, Hansen (
#'  2014, ReStud), Chernozhukov, Hansen, Spindler (2015, Annual Review of Economics)	
#' 	
#' - Here we can do other principled model selection or regularization methods more generally.	
#' 	
#' ---	
#' 	
#' - Put results in the table	
	

library(xtable)	
table<- matrix(0, 3, 2)	
table[1,1]<- summary(full.fit)$coef["female",1]   #extract coeff on female indicator  
table[1,2]<- full.se["female"]   #extract robust se on female indicator  
table[2,1]<- summary(partial.fit)$coef["rD",1]   #extract coeff on female indicator  
table[2,2]<- partial.se["rD"]   #extract robust se on female indicator  
table[3,1]<- summary(partial.fit.lasso)$coef["rD",][1]	
table[3,2]<- partial.lasso.se["rD"]   #extract robust se on female indicator  
colnames(table)<- names(summary(full.fit)$coef["female",])[1:2]	
rownames(table)<- c("full reg", "partial reg", "partial reg via lasso")	
tab<- xtable(table, digits=c(3, 3, 10))	
#' 	

print(tab, type="latex");

