## ----include=FALSE------------------------------------------------------------
##### _Advanced Data Analysis from an Elementary Point of View_ #####
# R code for the chapter "Density Estimation"
# Please do not re-distribute or use without attribution
# http://www.stat.cmu.edu/~cshalizi/ADAfaEPoV/




## ----include=FALSE------------------------------------------------------------
library(np) # Load the np library, but do it in a code block which doesn't
            # write out any extraneous messages


## ----popinv-joint-pdf, echo=FALSE---------------------------------------------
# Fit the joint density of logarithmic population growth and investment
# rates
data(oecdpanel)
popinv <- npudens(~popgro+inv, data=oecdpanel)
# Call the np plotting routine, but just ask it to calculate, rather than to
# make a plot --- see help(npplot)
fhat <- plot(popinv,plot.behavior="data")$d1
  # Now fhat$eval contains the grid of evaluation points, and fhat$dens the
  # actual density values
# Load a useful graphics library
library(lattice)
# Make a contour plot of the joint density, with more levels, and smaller
# labels on the levels, than usual
contourplot(fhat$dens~fhat$eval$Var1*fhat$eval$Var2,cuts=20,
  xlab="popgro",ylab="inv",labels=list(cex=0.5))



## ----popgro-vs-year-pdf, echo=FALSE-------------------------------------------
# Fit the density of logarithmic population growth rates conditional on year
pop.cdens <- npcdens(popgro ~ year,data=oecdpanel)
# Manually construct a grid of points on which to plot
plotting.grid <- expand.grid(year=seq(from=1965,to=1995,by=1),
  popgro=seq(from=-3.5,to=-2.4,length.out=300))
# Evaluate the function
fhat <- predict(pop.cdens,newdata=plotting.grid)
# Make a wireframe plot
wireframe(fhat~plotting.grid$year*plotting.grid$popgro,
  scales=list(arrows=FALSE),xlab="year",ylab="popgro",zlab="pdf")



## ----popgro-vs-year-vs-oecd-pdf,echo=FALSE------------------------------------
# Fit the density of popgro conditional on year and OECD membership
pop.cdens.o <- npcdens(popgro~year+factor(oecd),data=oecdpanel)
  # Ensure that npcdens treats oecd as a categorical variable, not a number
# Make the grid, but now include the OECD factor
oecd.grid <- expand.grid(year=seq(from=1965,to=1995,by=1),
  popgro=seq(from=-3.4,to=-2.4,length.out=300),
  oecd=unique(oecdpanel$oecd))
fhat <- predict(pop.cdens.o,newdata=oecd.grid)
# Make side-by-side wireframes, conditional on OECD membership
wireframe(fhat~oecd.grid$year*oecd.grid$popgro|oecd.grid$oecd,
  scales=list(arrows=FALSE),xlab="year",ylab="popgro",zlab="pdf")



## ----popinv-exponentiated-joint-pdf, echo=FALSE-------------------------------
popinv2 <- npudens(~exp(popgro)+exp(inv),data=oecdpanel)
# Plotting code comitted



## -----------------------------------------------------------------------------
rpopinv <- function(n) {
  n.train <- length(popinv2$dens)
  ndim <- popinv2$ndim
  points <- sample(1:n.train,size=n,replace=TRUE)
  z <- matrix(0,nrow=n,ncol=ndim)
  for (i in 1:ndim) {
    coordinates <- popinv2$eval[points,i]
    z[,i] <- rnorm(n,coordinates,popinv2$bw[i])
  }
  colnames(z) <- c("pop.growth.rate","invest.rate")
  return(z)
}


## -----------------------------------------------------------------------------
signif(mean(exp(oecdpanel$popgro)),3)
signif(mean(exp(oecdpanel$inv)),3)
signif(colMeans(rpopinv(200)),3)


## -----------------------------------------------------------------------------
z <- rpopinv(2000)
signif(mean(z[,"invest.rate"]/z[,"pop.growth.rate"]),3)
signif(sd(z[,"invest.rate"]/z[,"pop.growth.rate"])/sqrt(2000),3)


## -----------------------------------------------------------------------------
signif(median(z[,"invest.rate"]/z[,"pop.growth.rate"]),3)


## ----dist-of-y-over-x, out.width="0.5\\textwidth", echo=FALSE-----------------
YoverX <- z[,"invest.rate"]/z[,"pop.growth.rate"]
plot(density(YoverX),xlab="Y/X",ylab="Probability density",main="")
rug(YoverX,side=1)



## ----many-dists-of-y-over-x, out.width="0.5\\textwidth", echo=FALSE-----------
plot(0,xlab="Y/X",ylab="Probability density",type="n",xlim=c(-1,10),
     ylim=c(0,0.3))
one.plot <- function() {
  zprime <- rpopinv(2000)
  YoverXprime <- zprime[,"invest.rate"]/zprime[,"pop.growth.rate"]
  density.prime <- density(YoverXprime)
  lines(density.prime,col="grey")
}
invisible(replicate(50,one.plot()))

