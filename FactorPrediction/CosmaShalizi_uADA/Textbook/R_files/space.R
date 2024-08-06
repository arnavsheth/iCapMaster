## ----include=FALSE------------------------------------------------------------
##### _Advanced Data Analysis from an Elementary Point of View_ #####
# R code for the chapter "Spatial Statistics"
# Please do not re-distribute or use without attribution
# http://www.stat.cmu.edu/~cshalizi/ADAfaEPoV/


## ----echo=FALSE---------------------------------------------------------------
library(knitr)
opts_chunk$set(size='small', background='white', cache=TRUE, autodep=TRUE,
               tidy=TRUE, tidy.opts=list(comment=FALSE), highlights=FALSE)
# opts_chunk$set(fig.path="space-and-nets/")


## -----------------------------------------------------------------------------
# Create the 9x2 matrix of coordinates
  # Slight notation clash to call the coordinates "x" and "y", but this
  # will simplify later plotting
coords <- expand.grid(x=c(-1,0,1), y=c(-1,0,1))
# Keep track of which row is the origin (where we want to predict at)
predict.pt <- which(coords$x==0 & coords$y==0)
# Use the built-in function to create distance matrices
distances <- dist(coords) # Euclidean matrix by default
# That returns a special structure w/ just lower-triangular half
distances <- as.matrix(distances)
# Create the matrix of all covariances
covars <- exp(-distances)
# Break it into Cov(Y,Z) and Var(Z)
Cov.YZ <- covars[predict.pt, - predict.pt]
Var.Z <- covars[-predict.pt, -predict.pt]
# Find the coefficients
beta <- solve(Var.Z) %*% Cov.YZ
# Print: for each coordinate, what is the coefficient?
signif(data.frame(coords[-predict.pt,], coef=beta),3)


## ----minimal-example, echo=FALSE----------------------------------------------
plot(coords, xlab="longitude", ylab="latitude", type="n")
points(coords[predict.pt,], col="red")
points(coords[-predict.pt,], cex=10*sqrt(beta))

## ----eval=FALSE---------------------------------------------------------------
## plot(coords, xlab="longitude", ylab="latitude", type="n")
## points(coords[predict.pt,], col="red")
## points(coords[-predict.pt,], cex=10*sqrt(beta))

