## ----components-of-generalization-error, echo=FALSE, out.width="0.5\\textwidth"----
curve(2*x^4,from=0,to=1,lty=2,xlab="Smoothing",ylab="Generalization error")
curve(0.12+x-x,lty=3,add=TRUE)
curve(1/(10*x),lty=4,add=TRUE)
curve(0.12+2*x^4+1/(10*x),add=TRUE)

## ----error-components-with-more-data, out.width="0.5\\textwidth", echo=FALSE----
curve(2*x^4,from=0,to=1,lty=2,xlab="Smoothing",ylab="Generalization error")
curve(0.12+x-x,lty=3,add=TRUE)
curve(1/(10*x),lty=4,add=TRUE,col="grey")
curve(0.12+2*x^2+1/(10*x),add=TRUE,col="grey")
curve(1/(30*x),lty=4,add=TRUE)
curve(0.12+2*x^4+1/(30*x),add=TRUE)

## ----two-curves, echo=FALSE----------------------------------------------
par(mfcol=c(2,1))
true.r <- function(x) {sin(x)*cos(20*x)}
true.s <- function(x) {log(x+1)}
curve(true.r(x),from=0,to=3,xlab="x",ylab=expression(f(x)))
curve(true.s(x),from=0,to=3,xlab="x",ylab=expression(g(x)))
par(mfcol=c(1,1))

## ----two-curves-noisy, echo=FALSE----------------------------------------
x = runif(300,0,3)
yr = true.r(x)+rnorm(length(x),0,0.15)
ys = true.s(x)+rnorm(length(x),0,0.15)
par(mfcol=c(2,1))
plot(x,yr,xlab="x",ylab=expression(r(x)+epsilon))
curve(true.r(x),col="grey",add=TRUE)
plot(x,ys,xlab="x",ylab=expression(s(x)+eta))
curve(true.s(x),col="grey",add=TRUE)

## ----local-averaging, echo=FALSE-----------------------------------------
par(mfcol=c(2,1))
x.focus <- 1.6; x.lo <- x.focus-0.1; x.hi <- x.focus+0.1
colors=ifelse((x<x.hi)&(x>x.lo),"black","grey")
plot(x,yr,xlab="x",ylab=expression(r(x)+epsilon),col=colors)
curve(true.r(x),col="grey",add=TRUE)
points(x.focus,mean(yr[(x<x.hi)&(x>x.lo)]),pch=18,cex=2)
plot(x,ys,xlab="x",ylab=expression(s(x)+eta),col=colors)
curve(true.s(x),col="grey",add=TRUE)
points(x.focus,mean(ys[(x<x.hi)&(x>x.lo)]),pch=18,cex=2)
par(mfcol=c(1,1))

## ----local-averaging-errors, out.width="0.4\\textwidth", echo=FALSE------
loc_ave_err <- function(h,y,y0) {abs(y0-mean(y[(x.focus-h < x) & (x.focus+h>x)]))}
yr0=true.r(x.focus); ys0=true.s(x.focus)
r.LAE = sapply(1:100/100,loc_ave_err,y=yr,y0=yr0)
s.LAE = sapply(1:100/100,loc_ave_err,y=ys,y0=ys0)
plot(1:100/100,r.LAE,xlab="Radius of averaging window",ylim=c(0,1.1),
     ylab="Absolute value of error",type="l",log="x")
lines(1:100/100,s.LAE,lty="dashed")
abline(h=0.15,col="grey")

## ------------------------------------------------------------------------
cv_bws_npreg <- function(x,y,bandwidths=(1:50)/50,nfolds=10) {
  require(np)
  n <- length(x)
  stopifnot(n > 1, length(y) == n)
  stopifnot(length(bandwidths) > 1)
  stopifnot(nfolds > 0, nfolds==trunc(nfolds))

  fold_MSEs <- matrix(0,nrow=nfolds,ncol=length(bandwidths))
  colnames(fold_MSEs) = bandwidths

  case.folds <- sample(rep(1:nfolds,length.out=n))
  for (fold in 1:nfolds) {
    train.rows = which(case.folds!=fold)
    x.train = x[train.rows]
    y.train = y[train.rows]
    x.test = x[-train.rows]
    y.test = y[-train.rows]
    for (bw in bandwidths) {
      fit <- npreg(txdat=x.train,tydat=y.train,
                   exdat=x.test,eydat=y.test,bws=bw)
      fold_MSEs[fold,paste(bw)] <- fit$MSE
    }
  }
  CV_MSEs = colMeans(fold_MSEs)
  best.bw = bandwidths[which.min(CV_MSEs)]
  return(list(best.bw=best.bw,CV_MSEs=CV_MSEs,fold_MSEs=fold_MSEs))
}

## ----RMS-error-vs-bandwiths, out.width="0.5\\textwidth", echo=FALSE, message=FALSE----
rbws <- cv_bws_npreg(x,yr,bandwidths=(1:100)/200)
sbws <- cv_bws_npreg(x,ys,bandwidths=(1:100)/200)
plot(1:100/200,sqrt(rbws$CV_MSEs),xlab="Bandwidth",
  ylab="Root CV MSE",type="l",ylim=c(0,0.6),log="x")
lines(1:100/200,sqrt(sbws$CV_MSEs),lty="dashed")
abline(h=0.15,col="grey")

## ----two-curves-plus-smooths, echo=FALSE, message=FALSE------------------
x.ord=order(x)
par(mfcol=c(2,1))
plot(x,yr,xlab="x",ylab=expression(r(x)+epsilon))
rhat <- npreg(bws=rbws$best.bw,txdat=x,tydat=yr)
lines(x[x.ord],fitted(rhat)[x.ord],lwd=4)
curve(true.r(x),col="grey",add=TRUE,lwd=2)
plot(x,ys,xlab="x",ylab=expression(s(x)+eta))
shat <- npreg(bws=sbws$best.bw,txdat=x,tydat=ys)
lines(x[x.ord],fitted(shat)[x.ord],lwd=4)
curve(true.s(x),col="grey",add=TRUE,lwd=2)
par(mfcol=c(1,1))

## ----demo-surface, echo=FALSE, out.width="0.5\\textwidth"----------------
x1.points <- seq(-3,3,length.out=100)
x2.points <- x1.points
x12grid <- expand.grid(x1=x1.points,x2=x2.points)
y <- matrix(0,nrow=100,ncol=100)
y <- outer(x1.points,x2.points,f)
library(lattice)
wireframe(y~x12grid$x1*x12grid$x2,scales=list(arrows=FALSE),
  xlab=expression(x^1),ylab=expression(x^2),zlab="y")

## ----demo-surface-plus-noise, echo=FALSE, out.width="0.5\\textwidth"-----
x1.noise <- runif(1000,min=-3,max=3)
x2.noise <- runif(1000,min=-3,max=3)
y.noise <- f(x1.noise,x2.noise)+rnorm(1000,0,0.05)
noise <- data.frame(y=y.noise,x1=x1.noise,x2=x2.noise)
cloud(y~x1*x2,data=noise,col="black",scales=list(arrows=FALSE),
      xlab=expression(x^1),ylab=expression(x^2),zlab="y")

## ----demo-surface-reconstruction, out.width="0.5\\textwidth", echo=FALSE----
noise.np <- npreg(y~x1+x2,data=noise)
y.out <- matrix(0,100,100)
y.out <- predict(noise.np,newdata=x12grid)
wireframe(y.out~x12grid$x1*x12grid$x2,scales=list(arrows=FALSE),
          xlab=expression(x^1),ylab=expression(x^2),zlab="y")

## ----cross-section, echo=FALSE, out.width="0.5\\textwidth"---------------
new.frame <- data.frame(x1=seq(-3,3,length.out=300),x2=median(y.noise))
plot(new.frame$x1,predict(noise.np,newdata=new.frame),
  type="l",xlab=expression(x^1),ylab="y",ylim=c(0,1.0))
new.frame$x2 <- quantile(y.noise,0.25)
lines(new.frame$x1,predict(noise.np,newdata=new.frame),lty=2)
new.frame$x2 <- quantile(y.noise,0.75)
lines(new.frame$x1,predict(noise.np,newdata=new.frame),lty=3)

## ----logistic-curve, out.width="0.5\\textwidth", echo=FALSE--------------
curve(exp(7*x)/(1+exp(7*x)),from=-5,to=5,ylab="y")

## ------------------------------------------------------------------------
make.demo.df <- function(n) {
  demo.func <- function(x,z,w) {
    20*x^2 + ifelse(w=="A",z,10*exp(z)/(1+exp(z)))
  }
  x <- runif(n,-1,1)
  z <- rnorm(n,0,10)
  w <- sample(c("A","B"),size=n,replace=TRUE)
  y <- demo.func(x,z,w)+rnorm(n,0,0.05)
  return(data.frame(x=x,y=y,z=z,w=w))
}
demo.df <- make.demo.df(100)

## ------------------------------------------------------------------------
demo.np1 <- npreg(y ~ x + z, data=demo.df)

## ------------------------------------------------------------------------
summary(demo.np1)

## ------------------------------------------------------------------------
demo.np1$MSE

## ------------------------------------------------------------------------
demo.np1$bws$fval

## ------------------------------------------------------------------------
predict(demo.np1, newdata=data.frame(x=-1,z=5))

## ------------------------------------------------------------------------
demo.np3 <- npreg(y~x+z+factor(w),data=demo.df)

## ------------------------------------------------------------------------
demo.np3$bws$fval

## ----npreg-demo-plot-1, out.width="0.5\\textwidth", echo=FALSE-----------
plot(demo.np1,theta=40,view="fixed")

## ----npreg-demo-plot-3, echo=FALSE---------------------------------------
plot(demo.np3)

## ------------------------------------------------------------------------
bigdemo.df <- make.demo.df(1e3)
system.time(demo.np4 <- npreg(y~x+z+factor(w), data=bigdemo.df,
                              tol=0.01,ftol=0.01))

## ------------------------------------------------------------------------
demo.np4$bws

## ----npreg-bigdemo-plot, echo=FALSE--------------------------------------
x.seq <- seq(from=-1,to=1,length.out=50)
z.seq <- seq(from=-30,to=30,length.out=50)
grid.A <- expand.grid(x=x.seq,z=z.seq,w="A")
grid.B <- expand.grid(x=x.seq,z=z.seq,w="B")
yhat.A <- predict(demo.np4,newdata=grid.A)
yhat.B <- predict(demo.np4,newdata=grid.B)
par(mfrow=c(1,2))
persp(x=x.seq,y=z.seq,z=matrix(yhat.A,nrow=50),theta=40,main="W=A",
  xlab="x",ylab="z",zlab="y",ticktype="detailed")
persp(x=x.seq,y=z.seq,z=matrix(yhat.B,nrow=50),theta=40,main="W=B",
  xlab="x",ylab="z",zlab="y",ticktype="detailed")

