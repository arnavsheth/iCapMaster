## ----include=FALSE------------------------------------------------------------
##### _Advanced Data Analysis from an Elementary Point of View_ #####
# R code for the chapter "Causal Discovery"
# Please do not re-distribute or use without attribution
# http://www.stat.cmu.edu/~cshalizi/ADAfaEPoV/




## ----results="hide"-----------------------------------------------------------
library(pcalg)
library(SMPracticals)
data(mathmarks)
suffStat <- list(C=cor(mathmarks),n=nrow(mathmarks))
pc.fit <- pc(suffStat, indepTest=gaussCItest, p=ncol(mathmarks),alpha=0.005)


## ----mathmarks-dag,echo=FALSE,results="hide",out.width="0.5\\textwidth"-------
library(Rgraphviz)
plot(pc.fit,labels=colnames(mathmarks),main="Inferred DAG for mathmarks")

## ----eval=FALSE---------------------------------------------------------------
## library(Rgraphviz)
## plot(pc.fit,labels=colnames(mathmarks),main="Inferred DAG for mathmarks")


## -----------------------------------------------------------------------------
summary(pc.fit)


## ----mathmarks-dag2,echo=FALSE,out.width="0.5\\textwidth"---------------------
plot(pc(suffStat, indepTest=gaussCItest, p=ncol(mathmarks),alpha=0.05),
  labels=colnames(mathmarks),main="")

## ----eval=FALSE---------------------------------------------------------------
## plot(pc(suffStat, indepTest=gaussCItest, p=ncol(mathmarks),alpha=0.05),
##   labels=colnames(mathmarks),main="")


## ----include=FALSE------------------------------------------------------------
psychs <- matrix(c(1.0, 0.62, 0.25, 0.16, -0.10, 0.29, 0.18,
               0.62, 1.0, 0.09, 0.28, 0.00, 0.25, 0.15,
               0.25, 0.09, 1.0, 0.07, 0.03, 0.34, 0.19,
               0.16, 0.28, 0.07, 1.0, 0.10, 0.37, 0.41,
               -0.10, 0.00, 0.03, 0.10, 1.0, 0.13, 0.43,
               0.29, 0.25, 0.34, 0.37, 0.13, 1.0, 0.55,
               0.18, 0.15, 0.19, 0.41, 0.43, 0.55, 1.0), ncol=7)
colnames(psychs) <- c("ability","grad","prod","first","sex","cites","pubs")
rownames(psychs) <- colnames(psychs)

## -----------------------------------------------------------------------------
psychs


## ----modeling-psychologists, echo=FALSE, out.width="0.5\\textwidth"-----------
plot(pc(list(C=psychs,n=162),indepTest=gaussCItest,p=7,alpha=0.01),
  labels=colnames(psychs),main="")

## ----eval=FALSE---------------------------------------------------------------
## plot(pc(list(C=psychs,n=162),indepTest=gaussCItest,p=7,alpha=0.01),
##   labels=colnames(psychs),main="")

