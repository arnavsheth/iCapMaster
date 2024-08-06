## ----include=FALSE------------------------------------------------------------
##### _Advanced Data Analysis from an Elementary Point of View_ #####
# R code for the chapter "Principal Components"
# Please do not re-distribute or use without attribution
# http://www.stat.cmu.edu/~cshalizi/ADAfaEPoV/




## ----geometry-of-pca----------------------------------------------------------
# Make an arc
demo.theta <- runif(10,min=0,max=pi/2)
demo.x <- cbind(cos(demo.theta),sin(demo.theta))
# Center the coordinates
demo.x <- scale(demo.x, center=TRUE, scale=FALSE)
# Plot the points
plot(demo.x,xlab=expression(x^1),ylab=expression(x^2), xlim=c(-1,1),
     ylim=c(-1,1))
# Pick a direction (not a very good one), by a unit vector
demo.w <- c(cos(-3*pi/8), sin(-3*pi/8))
# Draw an arrow for the unit vector
arrows(0,0,demo.w[1],demo.w[2],col="blue")
text(demo.w[1],demo.w[2],pos=4,labels=expression(w))
# Draw a dashed line for the whole line
abline(0,b=demo.w[2]/demo.w[1],col="blue",lty="dashed")
# Get the length of the projection of each data point on to the line
projection.lengths <- demo.x %*% demo.w
# Get the actual projections
projections <- projection.lengths %*% demo.w
# Draw those as points along the line
points(projections,pch=16,col="blue")
# Draw lines from each data point to its projection
segments(x0=demo.x[,1], y0=demo.x[,2],x1=projections[,1],y1=projections[,2], col="grey")


## -----------------------------------------------------------------------------
cars04 = read.csv("http://www.stat.cmu.edu/~cshalizi/ADAfaEPoV/data/cars-fixed04.dat")


## ----table_cars, echo=FALSE---------------------------------------------------
knitr(head(cars))


## -----------------------------------------------------------------------------
cars04.pca = prcomp(cars04[,8:18], scale.=TRUE)


## -----------------------------------------------------------------------------
round(cars04.pca$rotation[,1:2],2)


## ----cars_biplot, echo=FALSE--------------------------------------------------
biplot(cars04.pca,cex=0.4)



## ----cars_screeplot, echo=FALSE, out.width="0.5\\textwidth"-------------------
plot(cars04.pca,type="l",main="")



## -----------------------------------------------------------------------------
state.pca <- prcomp(state.x77,scale.=TRUE)


## ----states_pca_biplot, echo=FALSE, out.width="0.48\\textwidth"---------------
biplot(state.pca,cex=c(0.5,0.75))

## ----states_pca_scree, echo=FALSE, out.width="0.45\\textwidth"----------------
plot(state.pca,type="l")



## ----include=FALSE------------------------------------------------------------
state.confed <- rep(0,times=50)
names(state.confed) <- rownames(state.x77)
state.confed[c("South Carolina", "Mississippi", "Florida", "Alabama",
               "Georgia", "Louisiana", "Texas", "Virginia", "Arkansas",
               "Tennessee", "North Carolina")] <- 1
state.slave <- state.confed
state.slave[c("Kentucky","Missouri","Maryland","Delaware","West Virginia")] <- 1


## -----------------------------------------------------------------------------
signif(state.pca$rotation[,1:2],2)


## ----states_pca_2_southernness, echo=FALSE------------------------------------
# function to plot the state abbrevations in position, with scaled sizes
  # Linearly scale the sizes from the given minimum to the maximum
# Inputs: vector of raw numbers, minimum size for plot,
  # maximum size
# Outputs: Rescaled sizes (invisible)
plot.states_scaled <- function(sizes,min.size=0.4,max.size=2,...) {
  plot(state.center,type="n",...)
  out.range = max.size - min.size
  in.range = max(sizes)-min(sizes)
  scaled.sizes = out.range*((sizes-min(sizes))/in.range)
  text(state.center,state.abb,cex=scaled.sizes + min.size)
  invisible(scaled.sizes)
}

plot.states_scaled(state.pca$x[,1],min.size=0.3,max.size=1.5,
                   xlab="longitude",ylab="latitude")



## ----include=FALSE------------------------------------------------------------
load("~/teaching/ADAfaEPoV/data/pca-examples.Rdata")


## -----------------------------------------------------------------------------
load("~/teaching/ADAfaEPoV/data/pca-examples.Rdata")
nyt.pca <- prcomp(nyt.frame[,-1])
nyt.latent.sem <- nyt.pca$rotation


## -----------------------------------------------------------------------------
signif(sort(nyt.latent.sem[,1],decreasing=TRUE)[1:30],2)
signif(sort(nyt.latent.sem[,1],decreasing=FALSE)[1:30],2)


## -----------------------------------------------------------------------------
signif(sort(nyt.latent.sem[,2],decreasing=TRUE)[1:30],2)
signif(sort(nyt.latent.sem[,2],decreasing=FALSE)[1:30],2)


## ----first-two-pcs-of-nyt, echo=FALSE-----------------------------------------
plot(nyt.pca$x[,1:2],
     pch=ifelse(nyt.frame[,"class.labels"]=="music","m","a"),
     col=ifelse(nyt.frame[,"class.labels"]=="music","blue","red"))

