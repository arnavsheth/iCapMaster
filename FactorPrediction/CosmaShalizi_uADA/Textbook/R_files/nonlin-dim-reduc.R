## ----include=FALSE------------------------------------------------------------
##### _Advanced Data Analysis from an Elementary Point of View_ #####
# R code for the chapter "Nonlinear Dimensionality Reduction"
# Please do not re-distribute or use without attribution
# http://www.stat.cmu.edu/~cshalizi/ADAfaEPoV/




## ----spiral-creation, echo=FALSE----------------------------------------------
x=matrix(c(exp(-0.2*(-(1:300)/10))*cos(-(1:300)/10),
           exp(-0.2*(-(1:300)/10))*sin(-(1:300)/10)),
         ncol=2)
plot(x)



## ----spiral-with-pca, echo=FALSE----------------------------------------------
fit.all <- prcomp(x)
approx.all <- fit.all$x[,1]%*%t(fit.all$rotation[,1])
plot(x,xlab=expression(x[1]),ylab=expression(x[2]))
points(approx.all,pch=4)



## ----pca-of-spiral, echo=FALSE------------------------------------------------
plot(fit.all$x[,1], ylab="Score on first principal component")



## ----arc-segment, echo=FALSE--------------------------------------------------
fit = prcomp(x[270:280,])
pca.approx = fit$x[,1]%*%t(fit$rotation[,1])+colMeans(x[270:280,])
plot(rbind(x[270:280,],pca.approx),type="n",
     xlab=expression(x[1]),ylab=expression(x[2]))
points(x[270:280,])
points(pca.approx,pch=4)



## -----------------------------------------------------------------------------
# Local linear embedding of data vectors
# Inputs: n*p matrix of vectors, number of dimensions q to find (< p),
  # number of nearest neighbors per vector, scalar regularization setting
# Calls: find.kNNs, reconstruction.weights, coords.from.weights
# Output: n*q matrix of new coordinates
lle <- function(x,q,k=q+1,alpha=0.01) {
  stopifnot(q>0, q<ncol(x), k>q, alpha>0) # sanity checks
  kNNs = find.kNNs(x,k) # should return an n*k matrix of indices
  w = reconstruction.weights(x,kNNs,alpha) # n*n weight  matrix
  coords = coords.from.weights(w,q) # n*q coordinate matrix
  return(coords)
}


## -----------------------------------------------------------------------------
# Find multiple nearest neighbors in a data frame
# Inputs: n*p matrix of data vectors, number of neighbors to find,
  # optional arguments to dist function
# Calls: smallest.by.rows
# Output: n*k matrix of the indices of nearest neighbors
find.kNNs <- function(x,k,...) {
  x.distances = dist(x,...) # Uses the built-in distance function
  x.distances = as.matrix(x.distances) # need to make it a matrix
  kNNs = smallest.by.rows(x.distances,k+1) # see text for +1
  return(kNNs[,-1]) # see text for -1
}


## -----------------------------------------------------------------------------
# Find the k smallest entries in each row of an array
# Inputs: n*p array, p >= k, number of smallest entries to find
# Output: n*k array of column indices for smallest entries per row
smallest.by.rows <- function(m,k) {
  stopifnot(ncol(m) >= k) # Otherwise "k smallest" is meaningless
  row.orders = t(apply(m,1,order))
  k.smallest = row.orders[,1:k]
  return(k.smallest)
}


## -----------------------------------------------------------------------------
(r <- matrix(c(7,3,2,4),nrow=2))
smallest.by.rows(r,1)
smallest.by.rows(r,2)


## -----------------------------------------------------------------------------
round(as.matrix(dist(x[1:5,])),2)
smallest.by.rows(as.matrix(dist(x[1:5,])),3)


## -----------------------------------------------------------------------------
find.kNNs(x[1:5,],2)


## -----------------------------------------------------------------------------
# Least-squares weights for linear approx. of data from neighbors
# Inputs: n*p matrix of vectors, n*k matrix of neighbor indices,
  # scalar regularization setting
# Calls: local.weights
# Outputs: n*n matrix of weights
reconstruction.weights <- function(x,neighbors,alpha) {
  stopifnot(is.matrix(x),is.matrix(neighbors),alpha>0)
  n=nrow(x)
  stopifnot(nrow(neighbors) == n)
  w = matrix(0,nrow=n,ncol=n)
  for (i in 1:n) {
    i.neighbors = neighbors[i,]
    w[i,i.neighbors] = local.weights(x[i,],x[i.neighbors,],alpha)
  }
  return(w)
}


## -----------------------------------------------------------------------------
# Calculate local reconstruction weights from vectors
# Inputs: focal vector (1*p matrix), k*p matrix of neighbors,
  # scalar regularization setting
# Outputs: length k vector of weights, summing to 1
local.weights <- function(focal,neighbors,alpha) {
  # basic matrix-shape sanity checks
  stopifnot(nrow(focal)==1,ncol(focal)==ncol(neighbors))
  # Should really sanity-check the rest (is.numeric, etc.)
  k = nrow(neighbors)
  # Center on the focal vector
  neighbors=t(t(neighbors)-focal) # exploits recycling rule, which
    # has a weird preference for columns
  gram = neighbors %*% t(neighbors)
  # Try to solve the problem without regularization
  weights = try(solve(gram,rep(1,k)))
    # The try() function tries to evaluate its argument and returns
    # the value if successful; otherwise it returns an error
    # message of class "try-error"
  if (identical(class(weights),"try-error")) {
    # Un-regularized solution failed, try to regularize
      # ATTN: It'd be better to examine the error, and check
      # if it's something regularization could fix, before
      # trying a regularized version of the problem
    weights = solve(gram+alpha*diag(k),rep(1,k))
  }
  # Enforce the unit-sum constraint
  weights = weights/sum(weights)
  return(weights)
}


## -----------------------------------------------------------------------------
matrix(mapply("*",local.weights(x[1,],x[2:3,],0.01),x[2:3,]),nrow=2)
colSums(matrix(mapply("*",local.weights(x[1,],x[2:3,],0.01),x[2:3,]),nrow=2))
colSums(matrix(mapply("*",local.weights(x[1,],x[2:3,],0.01),x[2:3,]),nrow=2)) - x[1,]


## -----------------------------------------------------------------------------
colSums(matrix(mapply("*",local.weights(x[1,],x[2:4,],0.01),x[2:4,]),nrow=3)) -x[1,]


## -----------------------------------------------------------------------------
x.2NNs <- find.kNNs(x,2)
x.2NNs[1,]
local.weights(x[1,],x[x.2NNs[1,],],0.01)
wts<-reconstruction.weights(x,x.2NNs,0.01)
sum(wts[1,] != 0)
all(rowSums(wts != 0)==2)
all(rowSums(wts) == 1)
summary(rowSums(wts))


## -----------------------------------------------------------------------------
sum(wts[1,]) == 1
sum(wts[1,])
sum(wts[1,]) - 1
summary(rowSums(wts)-1)


## -----------------------------------------------------------------------------
all.equal(rowSums(wts), rep(1, ncol(wts)))


## -----------------------------------------------------------------------------
# Get approximation weights from indices of point and neighbors
# Inputs: index of focal point, n*p matrix of vectors, n*k matrix
  # of nearest neighbor indices, scalar regularization setting
# Calls: local.weights
# Output: vector of n reconstruction weights
local.weights.for.index <- function(focal,x,NNs,alpha) {
  n = nrow(x)
  stopifnot(n> 0, 0 < focal, focal <= n, nrow(NNs)==n)
  w = rep(0,n)
  neighbors = NNs[focal,]
  wts = local.weights(x[focal,],x[neighbors,],alpha)
  w[neighbors] = wts
  return(w)
}


## -----------------------------------------------------------------------------
w.1 = local.weights.for.index(1,x,x.2NNs,0.01)
w.1[w.1 != 0]
which(w.1 != 0)


## -----------------------------------------------------------------------------
# Local linear approximation weights, without iteration
# Inputs: n*p matrix of vectors, n*k matrix of neighbor indices,
  # scalar regularization setting
# Calls: local.weights.for.index
# Outputs: n*n matrix of reconstruction weights
reconstruction.weights.2 <- function(x,neighbors,alpha) {
  # Sanity-checking should go here
  n = nrow(x)
  w = sapply(1:n, local.weights.for.index, x=x, NNs=neighbors,
             alpha=alpha)
  w = t(w) # sapply returns the transpose of the matrix we want
  return(w)
}


## -----------------------------------------------------------------------------
wts.2 = reconstruction.weights.2(x,x.2NNs,0.01)
identical(wts.2,wts)


## -----------------------------------------------------------------------------
# Find intrinsic coordinates from local linear approximation weights
# Inputs: n*n matrix of weights, number of dimensions q, numerical
  # tolerance for checking the row-sum constraint on the weights
# Output: n*q matrix of new coordinates on the manifold
coords.from.weights <- function(w,q,tol=1e-7) {
  n=nrow(w)
  stopifnot(ncol(w)==n) # Needs to be square
  # Check that the weights are normalized
    # to within tol > 0 to handle round-off error
  stopifnot(all(abs(rowSums(w)-1) < tol))
  # Make the Laplacian
  M = t(diag(n)-w)%*%(diag(n)-w)
    # diag(n) is n*n identity matrix
  soln = eigen(M) # eigenvalues and eigenvectors (here,
    # eigenfunctions), in order of decreasing eigenvalue
  coords = soln$vectors[,((n-q):(n-1))] # bottom eigenfunctions
    # except for the trivial one
  return(coords)
}


## -----------------------------------------------------------------------------
spiral.lle = coords.from.weights(wts,1)
plot(spiral.lle,ylab="Coordinate on manifold")
all(diff(spiral.lle) > 0)


## ----lle-coordinate,echo=FALSE------------------------------------------------
plot(coords.from.weights(wts,1),ylab="Coordinate on manifold")



## ----spiral-rainbow,echo=FALSE------------------------------------------------
plot(x,col=rainbow(300,end=5/6)[cut(spiral.lle,300,labels=FALSE)])



## -----------------------------------------------------------------------------
all.equal(lle(x,1,2), spiral.lle)

