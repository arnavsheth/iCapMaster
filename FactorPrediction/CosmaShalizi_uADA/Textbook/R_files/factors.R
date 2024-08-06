## ----include=FALSE------------------------------------------------------------
##### _Advanced Data Analysis from an Elementary Point of View_ #####
# R code for the chapter "Factor Analysis"
# Please do not re-distribute or use without attribution
# http://www.stat.cmu.edu/~cshalizi/ADAfaEPoV/




## ----geometry-of-factors, echo=FALSE------------------------------------------
n <- 20; library(scatterplot3d)
f <- matrix(sort(rnorm(n)),ncol=1); w <- matrix(c(0.5,0.2,-0.1),nrow=1)
fw <- f %*% w; x <- fw + matrix(rnorm(n*3,sd=c(.15,.05,.09)),ncol=3,byrow=TRUE)
s3d <- scatterplot3d(x,xlab=expression(x^1),ylab=expression(x^2),
                     zlab=expression(x^3),pch=16)
s3d$points3d(matrix(seq(from=min(f)-1,to=max(f)+1,length.out=2),ncol=1)%*%w,
             col="red",type="l")
s3d$points3d(fw,col="red",pch=16)
for (i in 1:nrow(x)) {
  s3d$points3d(x=c(x[i,1],fw[i,1]),y=c(x[i,2],fw[i,2]),z=c(x[i,3],fw[i,3]),
               col="grey",type="l") }

## ----eval=FALSE---------------------------------------------------------------
## n <- 20; library(scatterplot3d)
## f <- matrix(sort(rnorm(n)),ncol=1); w <- matrix(c(0.5,0.2,-0.1),nrow=1)
## fw <- f %*% w; x <- fw + matrix(rnorm(n*3,sd=c(.15,.05,.09)),ncol=3,byrow=TRUE)
## s3d <- scatterplot3d(x,xlab=expression(x^1),ylab=expression(x^2),
##                      zlab=expression(x^3),pch=16)
## s3d$points3d(matrix(seq(from=min(f)-1,to=max(f)+1,length.out=2),ncol=1)%*%w,
##              col="red",type="l")
## s3d$points3d(fw,col="red",pch=16)
## for (i in 1:nrow(x)) {
##   s3d$points3d(x=c(x[i,1],fw[i,1]),y=c(x[i,2],fw[i,2]),z=c(x[i,3],fw[i,3]),
##                col="grey",type="l") }


## -----------------------------------------------------------------------------
(state.fa1 <- factanal(state.x77,factors=1,scores="regression"))


## ----include=FALSE------------------------------------------------------------
state.pca <- prcomp(state.x77,scale.=TRUE)
class(state.pca$rotation) <- "loadings"


## ----echo=FALSE---------------------------------------------------------------
print(state.pca$rotation)


## ----include=FALSE------------------------------------------------------------
# See fig:states_pca_2_southernness in the PCA chapter for this function
plot.states_scaled <- function(sizes,min.size=0.4,max.size=2,...) {
  plot(state.center,type="n",...)
  out.range = max.size - min.size
  in.range = max(sizes)-min(sizes)
  scaled.sizes = out.range*((sizes-min(sizes))/in.range)
  text(state.center,state.abb,cex=scaled.sizes + min.size)
  invisible(scaled.sizes)
}

## ----states_fa_1_southernness, echo=FALSE-------------------------------------
plot.states_scaled(state.fa1$score[,1],min.size=0.3,max.size=1.5,
                   xlab="longitude",ylab="latitude")

## ----eval=FALSE---------------------------------------------------------------
## plot.states_scaled(state.fa1$score[,1],min.size=0.3,max.size=1.5,
##                    xlab="longitude",ylab="latitude")


## -----------------------------------------------------------------------------
pvalues <- sapply(1:4,function(q){factanal(state.x77,factors=q)$PVAL})
signif(pvalues,2)


## ----pvalue-vs-number-of-factors, echo=FALSE, out.width="0.5\\textwidth"------
plot(1:4,pvalues,xlab="q (number of factors)", ylab="pvalue",
     log="y",ylim=c(1e-11,0.04))
abline(h=0.05,lty="dashed")

## ----eval=FALSE---------------------------------------------------------------
## plot(1:4,pvalues,xlab="q (number of factors)", ylab="pvalue",
##      log="y",ylim=c(1e-11,0.04))
## abline(h=0.05,lty="dashed")


## -----------------------------------------------------------------------------
print(factanal(state.x77, factors=4)$loadings)


## ----tidy=TRUE, tidy.opts=list(keep.comment=FALSE)----------------------------
# Simulate Godfrey Thomson's "sampling model" of mental abilities, and perform
# factor analysis on the resulting test scores.

# Simulate the Thomson model
  # Follow Thomson's original sampling-without-replacement scheme
  # Pick a random number in 1:a for the number of shared abilities for each test
  # Then draw a sample-without-replacement of that size from 1:a; those are the
  # shared abilities summed in that test.
  # Specific variance of each test is also random; draw a number in 1:q, and
  # sum that many independent normals, with the same parameters as the
  # abilities.
# Inputs: number of testees (n)
  # number of tests (d)
  # number of shared abilities (a)
  # number of specific abilities per test (q)
  # mean of each ability (mean)
  # sd of each ability (sd)
# Depends on: mvrnorm from library MASS (multivariate random normal generator)
# Output: list, containing:
  # matrix of test loadings on to general abilities
  # vector of number of specific abilities per test
  # matrix of abilities-by-testees
  # matrix of general+specific scores by testees
  # raw data (including measurement noise)
rthomson <- function(n,d,a,q,ability.mean=0,ability.sd=1) {
    # ATTN: Should really use more intuitive argument names
    # number of testees = n
    # number of tests = d
    # number of shared abilities = a
    # max. number of specific abilities per test = q

    stopifnot(require(MASS)) # for multivariate normal generation

    # assign abilities to tests
    general.per.test <- sample(1:a, size=d, replace=TRUE)
    specifics.per.test <- sample(1:q, size=d, replace=TRUE)

    # Define the matrix assigning abilities to tests
    general.to.tests <- matrix(0,a,d)
    # Exercise to the reader: Vectorize this
    for (i in 1:d) {
        abilities <- sample(1:a,size=general.per.test[i], replace=FALSE)
        general.to.tests[abilities,i] <- 1
    }

    # Covariance matrix of the general abilities
    sigma <- matrix(0,a,a)
    diag(sigma) <- (ability.sd)^2
    mu <- rep(ability.mean,a)
    x <- mvrnorm(n,mu,sigma) # person-by-abilities matrix of abilities

    # The "general" part of the tests
    general.tests <- x %*% general.to.tests
    # Now the "specifics"
    specific.tests <- matrix(0,n,d)
    noisy.tests <- matrix(0,n,d)
    # Each test gets its own specific abilities, which are independent for each
    # person
    # Exercise to the reader: vectorize this, too
    for (i in 1:d) {
        # Each test has noises.per.test disturbances, each of which has the
        # given sd; since these are all independent their variances add
        j <- specifics.per.test[i]
        specifics <- rnorm(n,mean=ability.mean*j,
                           sd=ability.sd*sqrt(j))
        specific.tests[,i] <- general.tests[,i] + specifics
        # Finally, for extra realism, some mean-zero trial-to-trial noise, so
        # that if we re-use this combination of general and specific ability
        # scores, we won't get the exact same test scores twice
        noises <- rnorm(n,mean=0,sd=ability.sd)
        noisy.tests[,i] <- specific.tests[,i] + noises
    }

    tm <- list(data=noisy.tests,
               general.ability.pattern = general.to.tests,
               numbers.of.specifics = specifics.per.test,
               ability.matrix = x,
              specific.tests = specific.tests)
    return(tm)
}


## -----------------------------------------------------------------------------
tm <- rthomson(50,11,500,50)
factanal(tm$data,1)


## ----thomson-model-pvalue-dist, echo=FALSE------------------------------------
plot(ecdf(replicate(200,factanal(rthomson(50,11,500,50)$data,1)$PVAL)),
       xlab="p value",ylab="Empirical CDF",
       main="Sampling distribution of FA p-value under Thomson model",
       sub="200 replicates of 50 subjects each")
abline(0,1,lty=2)

## ----eval=FALSE---------------------------------------------------------------
## NA

