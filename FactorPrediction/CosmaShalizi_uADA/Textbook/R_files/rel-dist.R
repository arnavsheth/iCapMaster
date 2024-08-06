## ----include=FALSE------------------------------------------------------------
##### _Advanced Data Analysis from an Elementary Point of View_ #####
# R code for the chapter "Relative Distribution Methods"
# Please do not re-distribute or use without attribution
# http://www.stat.cmu.edu/~cshalizi/ADAfaEPoV/




## ----neyman-smooth-functions,echo=FALSE---------------------------------------
par(mfrow=c(2,1))
h1 <- function(y) { sqrt(12)*(y-0.5) }
h2 <- function(y) { sqrt(5)*(6*(y-0.5)^2-0.5) }
h3 <- function(y) { sqrt(7)*(20*(y-0.5)^3 - 3*(y-0.5)) }
curve(h1(x),ylab=expression(h[j](y)),xlab="y")
curve(h2(x),add=TRUE,lty="dashed")
curve(h3(x),add=TRUE,lty="dotted")
legend(legend=c(expression(h[1]),expression(h[2]),expression(h[3])),
  lty=c("solid","dashed","dotted"),x="bottomright")
curve(exp(h1(x)),ylab=expression(e^h[j](y)),xlab="y")
curve(exp(h2(x)),add=TRUE,lty="dashed")
curve(exp(h3(x)),add=TRUE,lty="dotted")
legend(legend=c(expression(h[1]),expression(h[2]),expression(h[3])),
  lty=c("solid","dashed","dotted"),x="bottomright")
par(mfrow=c(1,1))



## ----smooth-alternative-vs-uniform,echo=FALSE---------------------------------
x <- (1:1e6)/1e6
z <- sum(exp(h1(x)+h2(x)-h3(x)))/1e6
curve(exp(h1(x)+h2(x)-h3(x))/z,xlab="y",ylab=expression(g(y,theta)))
abline(h=1,col="grey")



## ----neyman-after-transformation,echo=FALSE-----------------------------------
par(mfrow=c(2,1))
curve(h1(pnorm(x)),xlab="x",ylab=expression(h[j](F(x))),from=-5,to=5,
  ylim=c(-3,3))
curve(h2(pnorm(x)),add=TRUE,lty="dashed")
curve(h3(pnorm(x)),add=TRUE,lty="dotted")
legend(legend=c(expression(h[1]),expression(h[2]),expression(h[3])),
  lty=c("solid","dashed","dotted"),x="bottomright")
curve(dnorm(x)*exp(h1(pnorm(x))+h2(pnorm(x))-h3(pnorm(x)))/z,xlab="x",
  ylab=expression(g[X](x,theta)),from=-5,to=5)
curve(dnorm(x),add=TRUE,col="grey")
par(mfrow=c(1,1))



## ----include=FALSE------------------------------------------------------------
# load the ddst library, but quietly
library(ddst)


## -----------------------------------------------------------------------------
r <- rnorm(100)
(r.normality <- ddst.norm.test(r))


## -----------------------------------------------------------------------------
(r.normality <- ddst.norm.test(r,compute.p=TRUE))


## -----------------------------------------------------------------------------
pchisq(r.normality$statistic,df=1,lower.tail=FALSE)


## ----transforming-a-standard-gaussian-sample, echo=FALSE----------------------
par(mfrow=c(2,1))
plot(hist(r,plot=FALSE),freq=FALSE,main="")
rug(r)
curve(dnorm(x),add=TRUE,col="grey")
rF <- pnorm(r,mean=mean(r),sd=sd(r))
plot(hist(rF,plot=FALSE),freq=FALSE,main="")
rug(rF)
abline(h=1,col="grey")
par(mfrow=c(1,1))



## -----------------------------------------------------------------------------
ng <- rt(100,df=5)
ddst.norm.test(ng,compute.p=TRUE)


## -----------------------------------------------------------------------------
mean(replicate(100,ddst.norm.test(rt(100,df=5),compute.p=TRUE)$p.value)<0.05)


## ----transforming-a-non-gaussian-sample, echo=FALSE---------------------------
plot(hist(ng, plot=FALSE), freq=FALSE, main="")
rug(ng)
curve(dnorm(x, mean=mean(ng), sd=sd(ng)), add=TRUE, col="grey")
ngF <- pnorm(r, mean=mean(ng), sd=sd(ng))
plot(hist(ngF, plot=FALSE), freq=FALSE, main="")
rug(ngF)
abline(h=1, col="grey")



## ----include=FALSE------------------------------------------------------------
# Load the data set, but don't clutter the text
n90 <- read.csv("http://www.stat.cmu.edu/~cshalizi/uADA/13/hw/07/n90_pol.csv")


## ----non-relative-brain-dists,echo=FALSE--------------------------------------
par(mfrow=c(2,1))
plot(density(n90$amygdala[n90$orientation>2]),main="",
  xlab="Adjusted amygdala volume")
lines(density(n90$amygdala[n90$orientation<3]),lty="dashed")
plot(density(n90$acc[n90$orientation<3]),lty="dashed",main="",
  xlab="Adjusted ACC volume")
lines(density(n90$acc[n90$orientation>2]))
par(mfrow=c(1,1))



## ----include=FALSE------------------------------------------------------------
# Load the reldist library, without printing out its welcome message
library(reldist)


## ----eval=FALSE---------------------------------------------------------------
## acc.rel <- reldist(y=n90$acc[n90$orientation<3],
##                    yo=n90$acc[n90$orientation>2], ci=TRUE,
##                    yolabs=pretty(n90$acc[n90$orientation>2]),
##                    main="Relative density of adjusted ACC volume")


## ----brain-rel-dists,echo=FALSE-----------------------------------------------
par(mfrow=c(2,1))
reldist(y=n90$amygdala[n90$orientation<3],
        yo=n90$amygdala[n90$orientation>2],
        ci=TRUE,
        yolabs=pretty(n90$amygdala[n90$orientation>2]),
        main="Relative density of adjusted amygdala volume")
reldist(y=n90$acc[n90$orientation<3],
        yo=n90$acc[n90$orientation>2],
        ci=TRUE,
        yolabs=pretty(n90$acc[n90$orientation>2]),
        main="Relative density of adjusted ACC volume")
par(mfrow=c(1,1))



## ----include=FALSE------------------------------------------------------------
library(np)
data(oecdpanel)
in.oecd <- oecdpanel$oecd==1
reldist(y=oecdpanel$growth[in.oecd],
        yo=oecdpanel$growth[!in.oecd],
        yolabs=pretty(oecdpanel$growth[!in.oecd]))


## ----relative-growth-dist,echo=FALSE------------------------------------------
reldist(y=oecdpanel$growth[in.oecd],
        yo=oecdpanel$growth[!in.oecd],
        yolabs=pretty(oecdpanel$growth[!in.oecd]),
        ci=TRUE,ylim=c(0,3))



## ----relative-dist-compared-to-a-gaussian,echo=FALSE,warning=FALSE------------
growth.mean <- mean(oecdpanel$growth[!in.oecd])
growth.sd <- sd(oecdpanel$growth[!in.oecd])
r = pnorm(oecdpanel$growth[in.oecd],growth.mean,growth.sd)
reldist(y=r,ci=TRUE,ylim=c(0,3))
top.ticks <- (1:9)/10
top.tick.values <- signif(qnorm(top.ticks,growth.mean,growth.sd),2)
axis(side=3,at=top.ticks,labels=top.tick.values)



## ----adjusted-growth-rel-dist,echo=FALSE--------------------------------------
reldist(y=oecdpanel$growth[in.oecd],
  yo=oecdpanel$growth[!in.oecd],
  yolabs=pretty(oecdpanel$growth[!in.oecd]),
  z=oecdpanel$humancap[in.oecd],
  zo=oecdpanel$humancap[!in.oecd],
  decomp="covariate",
  ci=TRUE,ylim=c(0,4))



## ----rel-dist-of-educ,echo=FALSE----------------------------------------------
reldist(y=exp(oecdpanel$humancap[in.oecd]),
  yo=exp(oecdpanel$humancap[!in.oecd]),
  yolabs=pretty(exp(oecdpanel$humancap[!in.oecd])))

