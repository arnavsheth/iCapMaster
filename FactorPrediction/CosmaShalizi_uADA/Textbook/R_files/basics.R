## ----include=FALSE------------------------------------------------------------
##### _Advanced Data Analysis from an Elementary Point of View_ #####
# R code for the chapter "Regression"
# Please do not re-distribute or use without attribution
# http://www.stat.cmu.edu/~cshalizi/ADAfaEPoV/






## ----data-scatterplot, echo=FALSE---------------------------------------------
# Plot the data
plot(all.x,all.y,xlab="x",ylab="y")
rug(all.x,side=1,col="grey")
rug(all.y,side=2,col="grey")



## ----data-with-mean,echo=FALSE------------------------------------------------
# Plot the data
plot(all.x,all.y,xlab="x",ylab="y")
rug(all.x,side=1,col="grey")
rug(all.y,side=2,col="grey")
# Add a line showing the global mean to the previous plot
abline(h=mean(all.y),lty="dotted")



## ----rapidly-varying-function,echo=FALSE, out.width="0.5\\textwidth", tidy=FALSE----
# Example of a function which varies rapidly but within narrow limits ---
  # we actually do better by fitting a constant than matching the correct
  # functional form!
ugly.func <- function(x) {1 + 0.01*sin(100*x)}
x <- runif(20); y <- ugly.func(x) + rnorm(length(x),0,0.5)
# Plot the data plus the true curve
plot(x, y, xlab="x", ylab="y"); curve(ugly.func,add=TRUE)
# Plot the global sample mean
abline(h=mean(y), col="red", lty="dashed")
# Fit the correct functional form
sine.fit = lm(y ~ 1+ sin(100*x))
# Plot the fitted function
curve(sine.fit$coefficients[1]+sine.fit$coefficients[2]*sin(100*x),
      col="blue", add=TRUE, lty="dotted")
legend("topright", legend=c(expression(1+0.1*sin(100*x)),
                            expression(bar(y)),
                            expression(hat(a) + hat(b)*sin(100*x))),
       lty=c("solid", "dashed", "dotted"), col=c("black", "red", "blue"))




## ----data-line,echo=FALSE-----------------------------------------------------
# Go back to the running example data
# Plot the data
plot(all.x,all.y,xlab="x",ylab="y")
rug(all.x,side=1,col="grey")
rug(all.y,side=2,col="grey")
# Add a line showing the global mean to the previous plot
abline(h=mean(all.y),lty="dotted")
# Fit a linear model
fit.all = lm(all.y~all.x)
# Add it to the plot
abline(fit.all)



## ----data-knn-reg,echo=FALSE--------------------------------------------------
# Plot the running data + line for global mean
# Plot the data
plot(all.x,all.y,xlab="x",ylab="y")
rug(all.x,side=1,col="grey")
rug(all.y,side=2,col="grey")
# Add a line showing the global mean to the previous plot
abline(h=mean(all.y),lty="dotted")
library(FNN) # For "fast nearest neighbors" methods
# Grid on which to plot fits: knn.reg expects a matrix
plot.seq <- matrix(seq(from=0,to=1,length.out=100), byrow=TRUE)
# kNN regression lines, k from 1 to 20
lines(plot.seq,knn.reg(train=all.x,test=plot.seq,y=all.y,k=1)$pred,col="red")
lines(plot.seq,knn.reg(train=all.x,test=plot.seq,y=all.y,k=3)$pred,col="green")
lines(plot.seq,knn.reg(train=all.x,test=plot.seq,y=all.y,k=5)$pred,col="blue")
lines(plot.seq,knn.reg(train=all.x,test=plot.seq,y=all.y,k=20)$pred,col="purple")
legend("center", legend=c("mean",expression(k==1),
                          expression(k==3),
                          expression(k==5),
                          expression(k==20)), lty=c("dashed",rep("solid",4)), col=c("black","red","green","blue","purple"))



## ----data-ksmooth,echo=FALSE--------------------------------------------------
# Plot the running data + line for global mean
# Plot the data
plot(all.x,all.y,xlab="x",ylab="y")
rug(all.x,side=1,col="grey")
rug(all.y,side=2,col="grey")
# Add a line showing the global mean to the previous plot
abline(h=mean(all.y),lty="dotted")
lines(ksmooth(all.x, all.y, "box", bandwidth=2),col="red")
lines(ksmooth(all.x, all.y, "box", bandwidth=1),col="green")
lines(ksmooth(all.x, all.y, "box", bandwidth=0.1),col="blue")
lines(ksmooth(all.x, all.y, "normal", bandwidth=2),col="red",lty="dashed")
lines(ksmooth(all.x, all.y, "normal", bandwidth=1),col="green",lty="dashed")
lines(ksmooth(all.x, all.y, "normal", bandwidth=0.1),col="blue",lty="dashed")
legend("bottom", ncol=3, legend=c("", expression(h==2),  expression(h==1),
                                  expression(h==0.1),
                                  "Box", "", "", "",
                                  "Gaussian", "", "", ""),
       lty=c("blank", "blank", "blank",  "blank",
             "blank", "solid", "solid", "solid",
             "blank", "dashed", "dashed", "dashed"),
       col=c("black", "black", "black", "black",
             "black", "red", "green", "blue",
             "black", "red", "green", "blue"),
       pch=NA)



