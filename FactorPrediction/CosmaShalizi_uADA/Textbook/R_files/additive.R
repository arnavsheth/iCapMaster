## ----include=FALSE------------------------------------------------------------
##### _Advanced Data Analysis from an Elementary Point of View_ #####
# R code for the chapter "Additive Models"
# Please do not re-distribute or use without attribution
# http://www.stat.cmu.edu/~cshalizi/ADAfaEPoV/




## ----converge-rates-illustrated, echo=FALSE, out.width="0.5\\textwidth"-------
curve(x^(-1),from=1,to=1e4,log="x",xlab="n",ylab="Excess MSE")
curve(x^(-4/5),add=TRUE,lty="dashed")
curve(x^(-1/26),add=TRUE,lty="dotted")
legend("topright",legend=c(expression(n^{-1}),
  expression(n^{-4/5}),expression(n^{-1/26})),
  lty=c("solid","dashed","dotted"))



## ----going-to-cali------------------------------------------------------------
housing <- read.csv("http://www.stat.cmu.edu/~cshalizi/ADAfaEPoV/data/calif_penn_2011.csv")
housing <- na.omit(housing)
calif <- housing[housing$STATEFP==6,]


## ----calif.lm-----------------------------------------------------------------
calif.lm <- lm(log(Median_house_value) ~ Median_household_income
  + Mean_household_income + POPULATION + Total_units + Vacant_units + Owners
  + Median_rooms + Mean_household_size_owners + Mean_household_size_renters
  + LATITUDE + LONGITUDE, data = calif)


## -----------------------------------------------------------------------------
print(summary(calif.lm),signif.stars=FALSE,digits=3)


## ----code:predlims------------------------------------------------------------
# Calculate rought +-2 standard deviation prediction limits
# Inputs: prediction object as returned by predict.lm (preds)
  # standard deviation around the regression function (sigma)
# Output: two-column array of lower and upper limits for each point
# Presumes: preds contains components named "fit" and "se.fit"
predlims <- function(preds, sigma) {
  prediction.sd <- sqrt(preds$se.fit^2+sigma^2)
  upper <- preds$fit+2*prediction.sd
  lower <- preds$fit-2*prediction.sd
  lims <- cbind(lower=lower,upper=upper)
  return(lims)
}




## ----linear-predictions, out.width="0.9\\textwidth", echo=FALSE---------------
plot(calif$Median_house_value,exp(preds.lm$fit),type="n",
  xlab="Actual price ($)",ylab="Predicted ($)", main="Linear model",
  ylim=c(0,exp(max(predlims.lm))))
segments(calif$Median_house_value,exp(predlims.lm[,"lower"]),
  calif$Median_house_value,exp(predlims.lm[,"upper"]), col="grey")
abline(a=0,b=1,lty="dashed")
points(calif$Median_house_value,exp(preds.lm$fit),pch=16,cex=0.1)



## ----include=FALSE------------------------------------------------------------
# Load the mgcv package, but do so in a code block which throws away all
# the output, so we don't bother with its start-up message
require(mgcv)


## ----calif.gam----------------------------------------------------------------
system.time(calif.gam <- gam(log(Median_house_value)
  ~ s(Median_household_income) + s(Mean_household_income) + s(POPULATION)
  + s(Total_units) + s(Vacant_units) + s(Owners) + s(Median_rooms)
  + s(Mean_household_size_owners) + s(Mean_household_size_renters)
  + s(LATITUDE) + s(LONGITUDE), data=calif))




## ----additive-predictions,echo=FALSE------------------------------------------
plot(calif$Median_house_value,exp(preds.gam$fit),type="n",
  xlab="Actual price ($)",ylab="Predicted ($)", main="First additive model",
  ylim=c(0,exp(max(predlims.gam))))
segments(calif$Median_house_value,exp(predlims.gam[,"lower"]),
  calif$Median_house_value,exp(predlims.gam[,"upper"]), col="grey")
abline(a=0,b=1,lty="dashed")
points(calif$Median_house_value,exp(preds.gam$fit),pch=16,cex=0.1)



## ----addfit-partial-responses,echo=FALSE--------------------------------------
plot(calif.gam,scale=0,se=2,shade=TRUE,pages=1)



## -----------------------------------------------------------------------------
calif.gam2 <- gam(log(Median_house_value)
  ~ s(Median_household_income) + s(Mean_household_income) + s(POPULATION)
  + s(Total_units) + s(Vacant_units) + s(Owners) + s(Median_rooms)
  + s(Mean_household_size_owners) + s(Mean_household_size_renters)
  + s(LONGITUDE,LATITUDE), data=calif)




## ----addfit2-partial-responses, echo=FALSE------------------------------------
plot(calif.gam2, scale=0, se=2, shade=TRUE, resid=TRUE, pages=1)



## ----interaction-plot-wireframe,echo=FALSE------------------------------------
plot(calif.gam2,select=10,phi=60, pers=TRUE,ticktype="detailed",cex.axis=0.5)



## ----interaction-plot-contour,echo=FALSE--------------------------------------
plot(calif.gam2,select=10,se=FALSE)



## -----------------------------------------------------------------------------
graymapper <- function(z, x=calif$LONGITUDE,y=calif$LATITUDE,
  n.levels=10,breaks=NULL,break.by="length",legend.loc="topright",
  digits=3,...) {
  my.greys = grey(((n.levels-1):0)/n.levels)
  if (!is.null(breaks)) {
    stopifnot(length(breaks) == (n.levels+1))
  }
  else {
    if(identical(break.by,"length")) {
      breaks = seq(from=min(z),to=max(z),length.out=n.levels+1)
    } else {
      breaks = quantile(z,probs=seq(0,1,length.out=n.levels+1))
    }
  }
  z = cut(z,breaks,include.lowest=TRUE)
  colors = my.greys[z]
  plot(x,y,col=colors,bg=colors,...)
  if (!is.null(legend.loc)) {
    breaks.printable <- signif(breaks[1:n.levels],digits)
    legend(legend.loc,legend=breaks.printable,fill=my.greys)
  }
  invisible(breaks)
}


## ----maps-of-predictions, echo=FALSE------------------------------------------
par(mfrow=c(2,2))
calif.breaks <- graymapper(calif$Median_house_value, pch=16, xlab="Longitude",
  ylab="Latitude",main="Data",break.by="quantiles")
graymapper(exp(preds.lm$fit), breaks=calif.breaks, pch=16, xlab="Longitude",
  ylab="Latitude",legend.loc=NULL, main="Linear model")
graymapper(exp(preds.gam$fit), breaks=calif.breaks, legend.loc=NULL,
  pch=16, xlab="Longitude", ylab="Latitude",main="First additive model")
graymapper(exp(preds.gam2$fit), breaks=calif.breaks, legend.loc=NULL,
  pch=16, xlab="Longitude", ylab="Latitude",main="Second additive model")
par(mfrow=c(1,1))



## ----maps-of-errors,echo=FALSE------------------------------------------------
par(mfrow=c(2,2))
graymapper(calif$Median_house_value, pch=16, xlab="Longitude",
  ylab="Latitude", main="Data", break.by="quantiles")
errors.in.dollars <- function(x) { calif$Median_house_value - exp(fitted(x)) }
lm.breaks <- graymapper(residuals(calif.lm), pch=16, xlab="Longitude",
  ylab="Latitude", main="Residuals of linear model",break.by="quantile")
graymapper(residuals(calif.gam), pch=16, xlab="Longitude",
  ylab="Latitude", main="Residuals errors of first additive model",
  breaks=lm.breaks, legend.loc=NULL)
graymapper(residuals(calif.gam2), pch=16, xlab="Longitude",
  ylab="Latitude", main="Residuals of second additive model",
  breaks=lm.breaks, legend.loc=NULL)
par(mfrow=c(1,1))

